import mmcv
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np

from mmcv.parallel import collate, scatter
from mmcv.ops import DynamicScatter
from mmdet.core import build_bbox_coder
from mmdet3d.models import build_model
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.models.builder import (BACKBONES, HEADS, NECKS)
from mmdet3d.core.bbox import get_box_type


class DynamicPillarFeatureNet(nn.Module):
    def __init__(self,
                 in_channels=4,
                 feat_channels=(64, ),
                 with_distance=False,
                 with_cluster_center=True,
                 with_voxel_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True,
                 input_norm=False):
        super().__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        if with_voxel_center:
            in_channels += 2
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self.fp16_enabled = False
        self.in_channels = in_channels
        self.input_norm = input_norm
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False),
                    norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)

        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.input_norm = input_norm

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0] + 1
        canvas_len = canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_channel, canvas_len)
        # Only include non-empty pillars
        indices = (
            voxel_coors[:, 0] * canvas_y * canvas_x +
            voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[:, indices.long()] = voxel_mean.t()

        # Step 2: get voxel mean for each point
        voxel_index = (
            pts_coors[:, 0] * canvas_y * canvas_x +
            pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        center_per_point = canvas[:, voxel_index.long()].t()
        return center_per_point

    def generate_voxel_feature(self, features, coors):
        features_ls = [features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 2))
            f_center[:, 0] = features[:, 0] - (
                coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                coors[:, 2].type_as(features) * self.vy + self.y_offset)
            # f_center[:, 2] = features[:, 2] - (
            #     coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        if self.input_norm:
            features[:, 0] = features[:, 0] / self.point_cloud_range[3] # x
            features[:, 1] = features[:, 1] / self.point_cloud_range[4] # y
            features[:, 2] = features[:, 2] / self.point_cloud_range[5] # z
            features[:, 3] = features[:, 3] / 255.0                     #intensity
        return features

    @force_fp32(out_fp16=True)
    def forward(self, features):
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features.squeeze(-1).squeeze(1))
        return point_feats


class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.pts_backbone = BACKBONES.build(cfg['pts_backbone'])
        self.pts_neck = NECKS.build(cfg['pts_neck'])
        self.cfg = cfg
        pts_bbox_head = cfg['pts_bbox_head']
        train_cfg = cfg['train_cfg']
        test_cfg = cfg['test_cfg']
        pts_train_cfg = train_cfg.pts if train_cfg else None
        pts_bbox_head.update(train_cfg=pts_train_cfg)
        pts_test_cfg = test_cfg.pts if test_cfg else None
        pts_bbox_head.update(test_cfg=pts_test_cfg)
        self.pts_bbox_head = HEADS.build(pts_bbox_head)
        self.bbox_coder = build_bbox_coder(cfg['pts_bbox_head']['bbox_coder'])
        self.box_code_size = self.bbox_coder.code_size

    def forward(self, x):
        x = self.pts_backbone(x)
        x = self.pts_neck(x)
        outs = self.pts_bbox_head(x)

        bbox_preds, scores, dir_scores = [], [], []
        for task_res in outs:
            bbox_preds.append(task_res[0]['reg'])
            bbox_preds.append(task_res[0]['height'])
            bbox_preds.append(task_res[0]['dim'])
            if 'vel' in task_res[0].keys():
                bbox_preds.append(task_res[0]['vel'])
            scores.append(task_res[0]['heatmap'])
            dir_scores.append(task_res[0]['rot'])
        bbox_preds = torch.cat(bbox_preds, dim=1)
        scores = torch.cat(scores, dim=1)
        dir_scores = torch.cat(dir_scores, dim=1)
        return scores, bbox_preds, dir_scores

    def get_boxes(self, cls_scores, bbox_preds, dir_scores, img_metas, rescale=False):
        rets = []
        scores_range = [0]
        bbox_range = [0]
        dir_range = [0]
        self.test_cfg = self.test_cfg['pts']
        for i, task_head in enumerate(self.task_heads):
            scores_range.append(scores_range[i] + self.num_classes[i])
            bbox_dim = int(bbox_preds[0].shape[1] / len(self.task_heads))
            bbox_range.append(bbox_range[i] + bbox_dim)
            dir_range.append(dir_range[i] + 2)
        for task_id in range(len(self.num_classes)):
            num_class_with_bg = self.num_classes[task_id]

            batch_heatmap = cls_scores[
                0][:, scores_range[task_id]:scores_range[task_id + 1],
                ...].sigmoid()

            batch_reg = bbox_preds[0][:,
                                    bbox_range[task_id]:bbox_range[task_id] + 2,
                                    ...]
            batch_hei = bbox_preds[0][:, bbox_range[task_id] +
                                    2:bbox_range[task_id] + 3, ...]

            batch_dim = torch.exp(bbox_preds[0][:, bbox_range[task_id] +
                                                3:bbox_range[task_id] + 6, ...])

            if bbox_dim == 8:
                batch_vel = bbox_preds[0][:, bbox_range[task_id] +
                                        6:bbox_range[task_id + 1], ...]
            else:
                batch_vel = None

            batch_rots = dir_scores[0][:,
                                    dir_range[task_id]:dir_range[task_id + 1],
                                    ...][:, 0].unsqueeze(1)
            batch_rotc = dir_scores[0][:,
                                    dir_range[task_id]:dir_range[task_id + 1],
                                    ...][:, 1].unsqueeze(1)

            temp = self.bbox_coder.decode(
                batch_heatmap,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id)
            assert self.test_cfg['nms_type'] in ['circle', 'rotate']
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            rets.append(
                self.pts_bbox_head.get_task_detections(
                    num_class_with_bg, batch_cls_preds, batch_reg_preds, 
                    batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
                    bboxes = img_metas[i]['box_type_3d'](bboxes,
                                                        self.bbox_coder.code_size)
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list


def parse_model(model):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())


def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])


def load_data(model, pcd):
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    data = dict(
        pts_filename=pcd,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # for ScanNet demo we need axis_align_matrix
        ann_info=dict(axis_align_matrix=np.eye(4)),
        sweeps=[],
        # set timestamp = 0
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device.index])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    return data


def build_pfn_model(cfg, checkpoint=None, device='cuda:0'):
    pts_voxel_encoder = DynamicPillarFeatureNet(
        in_channels=cfg.model['pts_voxel_encoder']['in_channels'],
        feat_channels=cfg.model['pts_voxel_encoder']['feat_channels'],
        with_distance=cfg.model['pts_voxel_encoder']['with_distance'],
        voxel_size=cfg.model['pts_voxel_encoder']['voxel_size'],
        point_cloud_range=cfg.model['pts_voxel_encoder']['point_cloud_range'],
        input_norm=cfg.model['pts_voxel_encoder']['input_norm'])
    pts_voxel_encoder.to(device).eval()
    checkpoint_pts_load = torch.load(checkpoint, map_location=device)
    dicts = {}
    for key in checkpoint_pts_load['state_dict'].keys():
        if 'pfn' in key:
            dicts[key.split('pts_voxel_encoder.')[1]
                  ] = checkpoint_pts_load['state_dict'][key]
    pts_voxel_encoder.load_state_dict(dicts)
    print('-----------------------')
    parse_model(pts_voxel_encoder)
    return pts_voxel_encoder


def build_backbone_model(cfg, checkpoint=None, device='cuda:0'):
    backbone = Backbone(cfg.model)
    backbone.to('cuda').eval()

    checkpoint = torch.load(checkpoint, map_location='cuda')
    dicts = {}
    for key in checkpoint["state_dict"].keys():
        if "backbone" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "neck" in key:
            dicts[key] = checkpoint["state_dict"][key]
        if "bbox_head" in key:
            dicts[key] = checkpoint["state_dict"][key]
    backbone.load_state_dict(dicts)

    torch.cuda.set_device(device)
    backbone.to(device)
    backbone.eval()
    return backbone


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    args = parser.parse_args()

    if isinstance(args.config, str):
        cfg = mmcv.Config.fromfile(args.config)
    elif not isinstance(args.config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(args.config)}')
    cfg.model.pretrained = None
    convert_SyncBN(cfg.model)
    cfg.model.train_cfg = None
    device = args.device

    # original model
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint_load = load_checkpoint(
            model, args.checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint_load['meta']:
            model.CLASSES = checkpoint_load['meta']['CLASSES']
        else:
            model.CLASSES = cfg.class_names
        if 'PALETTE' in checkpoint_load['meta']:  # 3D Segmentor
            model.PALETTE = checkpoint_load['meta']['PALETTE']
    model.cfg = cfg  # save the config in the model for convenience
    torch.cuda.set_device(device)
    model.to(device)
    model.eval()
    # parse_model(model)

    LOG_PATH = '/media/Data/caros/baidu/robobus/perception-explorer/output/'

    # load data
    data = load_data(model, args.pcd)
    pts = data['points'][0]
    voxels, coors = model.voxelize(pts)

    # ---------------- voxel_feature -------------
    pts_voxel_encoder = build_pfn_model(cfg, args.checkpoint, device=device)
    voxel_feature_pytorch = pts_voxel_encoder.generate_voxel_feature(voxels, coors)
    voxel_feature_trt = np.loadtxt(LOG_PATH + '00_voxel_feature.txt', dtype='float32')
    diff = np.abs(voxel_feature_pytorch - voxel_feature_trt)
    print(f"\nvoxel_feature shape:{diff.shape} with L1 diff {np.sum(diff)}, min diff:{np.min(diff)}, max diff:{np.max(diff)}")


    # ------------------- Pfe --------------------
    # point_size * 1 * 9 * 1
    voxel_feature_trt = voxel_feature_trt.reshape(voxel_feature_trt.shape[0], 1, pts_voxel_encoder.in_channels, 1)
    voxel_feature_trt = torch.from_numpy(voxel_feature_trt).to(device)
    pillar_feature = pts_voxel_encoder.forward(voxel_feature_trt)
    pillar_feature_pytorch = pillar_feature.cpu().detach().numpy()
    pillar_feature_trt = np.loadtxt(LOG_PATH + '03_pillar_feature.txt', dtype='float32')
    diff = np.abs(pillar_feature_pytorch - pillar_feature_trt)
    print(f"\npillar_feature shape:{diff.shape} with L1 diff {np.sum(diff)}, min diff:{np.min(diff)}, max diff:{np.max(diff)}")


    # ------------------- Scatter --------------------
    batch_size = coors[-1, 0] + 1
    canvas_feature = model.pts_middle_encoder(pillar_feature, coors, batch_size)
    
    canvas_feature_trt = np.loadtxt(
        LOG_PATH + '04_canvas_feature.txt', dtype='float32').reshape(1, 64, 800, 800)
    diff = np.abs(canvas_feature.cpu().detach().numpy() - canvas_feature_trt)
    print(f"\ncanvas_feature shape:{diff.shape} with L1 diff {np.sum(diff)}, min diff:{np.min(diff)}, max diff:{np.max(diff)}")


    canvas_feature = torch.from_numpy(canvas_feature_trt).to(device)

    # ------------------ Backbone ---------------------
    backbone_model = build_backbone_model(cfg, args.checkpoint, device)


    # ------------------ get_boxes -------------------
    from mmdet3d.core import bbox3d2result
    from mmdet3d.apis import show_result_meshlab

    img_metas = data['img_metas'][0]
    with torch.no_grad():
        scores, bbox_preds, dir_scores = backbone_model.forward(canvas_feature)

        bbox_preds_trt = np.loadtxt(LOG_PATH + '05_bbox_preds.txt', dtype='float32').reshape(1, 18, 200, 200)
        scores_trt = np.loadtxt(LOG_PATH + '06_cls_scores.txt', dtype='float32').reshape(1, 4, 200, 200)
        dir_scores_trt = np.loadtxt(LOG_PATH + '07_dir_scores.txt', dtype='float32').reshape(1, 6, 200, 200)

        diff = np.abs(bbox_preds.cpu().detach().numpy() - bbox_preds_trt)
        print(f"\nbbox_preds shape:{diff.shape} with L1 diff {np.sum(diff)}, min diff:{np.min(diff)}, max diff:{np.max(diff)}")
        diff = np.abs(scores.cpu().detach().numpy() - scores_trt)
        print(f"\nscores shape:{diff.shape} with L1 diff {np.sum(diff)}, min diff:{np.min(diff)}, max diff:{np.max(diff)}")
        diff = np.abs(dir_scores.cpu().detach().numpy() - dir_scores_trt)
        print(f"\ndir_scores shape:{diff.shape} with L1 diff {np.sum(diff)}, min diff:{np.min(diff)}, max diff:{np.max(diff)}")

        scores = torch.from_numpy(scores_trt).to(device)
        bbox_preds = torch.from_numpy(bbox_preds_trt).to(device)
        dir_scores = torch.from_numpy(dir_scores_trt).to(device)

        bbox_list = backbone_model.get_boxes([scores], [bbox_preds], [dir_scores], img_metas, True)
        bbox_pts = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        result = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(result, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
    
    show_result_meshlab(
        data,
        result,
        'tmp',
        None, 
        score_thr=0.1,
        show=args.show,
        snapshot=False,
        task='det',
        visualizer=None)


if __name__ == '__main__':
    main()
