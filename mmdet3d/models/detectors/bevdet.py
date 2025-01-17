# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import torch
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet3d.core import (Box3DMode, Coord3DMode, show_result)
from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint


@DETECTORS.register_module()
class BEVDet(CenterPoint):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        view_transformer (dict): Configuration dict of view transformer.
        bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """

    def __init__(self, view_transformer, bev_encoder_backbone,
                 bev_encoder_neck, **kwargs):
        super(BEVDet, self).__init__(**kwargs)
        self.view_transformer = builder.build_neck(view_transformer)
        self.bev_encoder_backbone = builder.build_backbone(
            bev_encoder_backbone)
        self.bev_encoder_neck = builder.build_neck(bev_encoder_neck)

    def image_encoder(self, imgs):
        """Image-view feature encoder."""
        B, N, C, H, W = imgs.shape
        imgs = imgs.view(B * N, C, H, W)
        x = self.img_backbone(imgs)
        if self.with_img_neck:
            x = self.img_neck(x)
        assert len(x) == 1
        _, out_C, out_H, out_W = x[0].shape
        x[0] = x[0].view(B, N, out_C, out_H, out_W)
        return x

    def bev_encoder(self, x):
        """Bird-Eye-View feature encoder."""
        x = self.bev_encoder_backbone(x)
        x = self.bev_encoder_neck(x)
        return x

    def extract_img_feat(self, img_inputs, img_metas):
        """Extract features of images."""
        x = self.image_encoder(img_inputs[0])
        x = self.view_transformer(x + img_inputs[1:])
        x = self.bev_encoder(x)
        return x

    def extract_feat(self, img_inputs, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img_inputs, img_metas)
        return img_feats

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img_inputs (list[torch.Tensor]): List inputs of BEVDet including
                Images of each sample with shape (N, C, H, W) and other
                matrixes for view transformation.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats = self.extract_feat(img_inputs, img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (torch.Tensor): Points is not necessary for camera-based
                BEVDet paradigm. Defaults to None.
            img_metas (list[dict]): Meta of each image in a batch.
            img_inputs (torch.Tensor): The inputs of BEVDet including Images
                of each sample with shape (N, C, H, W) and other matrixes for
                view transformation.
        """

        return self.simple_test(img_metas, img_inputs, **kwargs)

    def simple_test(self, img_metas, img_inputs, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img_inputs, img_metas)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        """Dummy forward function."""
        img_feats, _ = self.extract_feat(img_inputs, img_metas)
        img_metas = [dict(box_type_3d=LiDARInstance3DBoxes)]
        bbox_list = [dict() for _ in range(1)]
        assert self.with_pts_bbox
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=False)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def show_results(self, data, result, out_dir, show=False, score_thr=0.0):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'], DC):
                points = data['points']._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'], torch.Tensor):
                points = data['points'][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'])} "
                           f'for visualization!')
            if isinstance(data['img_metas'], DC):
                pts_filename = data['img_metas']._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas']._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'], dict):
                pts_filename = data['img_metas'][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name, show=show)
