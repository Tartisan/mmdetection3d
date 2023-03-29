import mmcv
from mmcv.cnn import build_norm_layer
from mmcv.runner import auto_fp16, force_fp32
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmcv.ops import DynamicScatter

import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from argparse import ArgumentParser

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
    checkpoint = torch.load(checkpoint, map_location=device)
    dicts = {}
    for key in checkpoint['state_dict'].keys():
        if 'pfn' in key:
            dicts[key.split('pts_voxel_encoder.')[1]
                  ] = checkpoint['state_dict'][key]
    pts_voxel_encoder.load_state_dict(dicts)
    return pts_voxel_encoder


def build_backbone_model(cfg, checkpoint=None, device='cuda:0'):
    backbone = Backbone(cfg.model)
    backbone.to(device).eval()

    checkpoint = torch.load(checkpoint, map_location=device)
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
        '--dynamic-axes',
        action='store_true',
        help='export onnx with dynamic_axes')
    parser.add_argument(
        '--show', action='store_true', help='show online visualization results')
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
    parse_model(model)

    data = load_data(model, args.pcd)
    pts = data['points'][0]
    voxels, coors = model.voxelize(pts)

    # DynamicPillarFeatureNet
    pts_voxel_encoder = build_pfn_model(cfg, args.checkpoint, device=device)
    dummy_input = torch.ones(160000, 1, pts_voxel_encoder.in_channels, 1).cuda()
    torch.onnx.export(pts_voxel_encoder,
                      dummy_input,
                      f='./tools/onnx_tools/cpdet/pfe.onnx',
                      opset_version=12,
                      verbose=True,
                      input_names=['voxels'],
                      output_names=['pillar_feature'],
                      dynamic_axes={'voxels': {0: 'point_size'},
                                    'pillar_feature': {0: 'point_size'}},
                      do_constant_folding=True)

    # Backbone
    voxel_features, feature_coors = model.pts_voxel_encoder(voxels, coors)
    batch_size = coors[-1, 0] + 1
    scattered_features = model.pts_middle_encoder(
        voxel_features, feature_coors, batch_size, None)
    backbone_model = build_backbone_model(cfg, args.checkpoint, device)
    torch.onnx.export(backbone_model,
                      scattered_features,
                      f='./tools/onnx_tools/cpdet/backbone.onnx',
                      opset_version=12,
                      verbose=True,
                      input_names=['canvas_feature'],
                      output_names=['scores', 'bbox_preds', 'dir_scores'],
                      do_constant_folding=True)


if __name__ == '__main__':
    main()
