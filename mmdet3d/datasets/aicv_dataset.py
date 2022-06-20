# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import tempfile
from os import path as osp
from copy import deepcopy

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log

from .builder import DATASETS
from ..core import show_multi_modality_result, show_result
from ..core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         LiDARInstance3DBoxes, points_cam2img)
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from ..core.evaluation.kitti_utils.eval import eval_class, get_mAP40


@DATASETS.register_module()
class AicvDataset(Custom3DDataset):
    r"""AICV Dataset.

    This class serves as the API for experiments on the `AICV Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        split (str): Split of input data.
        pts_prefix (str, optional): Prefix of points files.
            Defaults to 'velodyne'.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (list, optional): The range of point cloud used to
            filter invalid predicted boxes.
            Default: [-85, -85, -5, 85, 85, 5].
    """
    CLASSES = ('Car', 'Cyclist', 'Pedestrian')

    def __init__(self,
                 data_root,
                 ann_file,
                 split,
                 pts_prefix='velodyne',
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.split = split
        self.root_split = os.path.join(self.data_root, split)
        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.pts_prefix = pts_prefix

    def _get_pts_filename(self, idx):
        """Get point cloud filename according to the given index.

        Args:
            index (int): Index of the point cloud file to get.

        Returns:
            str: Name of the point cloud file.
        """
        pts_filename = osp.join(self.root_split, self.pts_prefix,
                                f'{idx:06d}.bin')
        return pts_filename

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - img_prefix (str): Prefix of image files.
                - img_info (dict): Image info.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['image']['image_idx']

        pts_filename = self._get_pts_filename(sample_idx)
        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            img_prefix=None)

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes.
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - difficulty (int): Difficulty defined by AICV.
                    0, 1, 2 represent xxxxx respectively.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]

        plane_lidar = None
        difficulty = info['annos']['difficulty']
        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)
        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)
        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_names = gt_names[selected]
        gt_bboxes = None

        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names,
            plane=plane_lidar,
            difficulty=difficulty)
        return anns_results

    def drop_arrays_by_name(self, gt_names, used_classes):
        """Drop irrelevant ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be dropped.
        """
        inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def keep_arrays_by_name(self, gt_names, used_classes):
        """Keep useful ground truths by name.

        Args:
            gt_names (list[str]): Names of ground truths.
            used_classes (list[str]): Classes of interest.

        Returns:
            np.ndarray: Indices of ground truths that will be keeped.
        """
        inds = [i for i, x in enumerate(gt_names) if x in used_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    def remove_dontcare(self, ann_info):
        """Remove annotations that do not need to be cared.

        Args:
            ann_info (dict): Dict of annotation infos. The ``'DontCare'``
                annotations will be removed according to ann_file['name'].

        Returns:
            dict: Annotations after filtering.
        """
        img_filtered_annotations = {}
        relevant_annotation_indices = [
            i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
        ]
        for key in ann_info.keys():
            img_filtered_annotations[key] = (
                ann_info[key][relevant_annotation_indices])
        return img_filtered_annotations

    def format_results(self,
                       outputs,
                       pklfile_prefix=None,
                       submission_prefix=None):
        """Format the results to pkl file.

        Args:
            outputs (list[dict]): Testing results of the dataset.
            pklfile_prefix (str): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str): The prefix of submitted files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing
                the json filepaths, tmp_dir is the temporal directory created
                for saving json files when jsonfile_prefix is not specified.
        """
        if pklfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        if 'pts_bbox' in outputs[0] or 'img_bbox' in outputs[0]:
            result_files = dict()
            for name in outputs[0]:
                results_ = [out[name] for out in outputs]
                pklfile_prefix_ = pklfile_prefix + name
                if submission_prefix is not None:
                    submission_prefix_ = submission_prefix + name
                else:
                    submission_prefix_ = None

                result_files_ = self.bbox2result_kitti(
                    results_, self.CLASSES, pklfile_prefix_,
                    submission_prefix_)
                result_files[name] = result_files_
        else:
            result_files = self.bbox2result_kitti(outputs, self.CLASSES,
                                                  pklfile_prefix,
                                                  submission_prefix)
        return result_files, tmp_dir

    def format_gt_annos(self):
        """
        dimensions: l, w, h -> l, h, w
        location:   x, y, z -> -y, -z, x
        rotation_y: r       -> -r - pi/2
        """
        gt_annos = []
        for info in self.data_infos:
            anno = deepcopy(info['annos'])
            # check box_preds
            valid_pcd_inds = ((anno['location'] > self.pcd_limit_range[:3]) &
                              (anno['location'] < self.pcd_limit_range[3:]))
            valid_inds = valid_pcd_inds.all(-1)
            for k, v in anno.items():
                anno[k] = anno[k][valid_inds]

            anno['location'] = (anno['location'] * [1, -1, -1])[:, [1, 2, 0]]
            anno['dimensions'] = anno['dimensions'][:, [0, 2, 1]]
            anno['rotation_y'] = -anno['rotation_y'] - np.pi / 2
            exceed_indx = np.where(anno['rotation_y'] < -np.pi)
            anno['rotation_y'][exceed_indx] = anno['rotation_y'][exceed_indx] + 2 * np.pi
            
            gt_annos.append(anno)
        return gt_annos

    def do_eval(self, 
                gt_annos,
                dt_annos,
                current_classes,
                min_overlaps,
                eval_types=['bev', '3d']):
        # min_overlaps: [num_minoverlap, metric, num_class]
        difficultys = [1]
        mAP40_bev = None
        if 'bev' in eval_types:
            ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                            min_overlaps)
            mAP40_bev = get_mAP40(ret['precision'])

        mAP40_3d = None
        if '3d' in eval_types:
            ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2,
                            min_overlaps)
            mAP40_3d = get_mAP40(ret['precision'])
        return (mAP40_bev, mAP40_3d)

    def aicv_eval(self, 
                  gt_annos,
                  dt_annos,
                  current_classes,
                  eval_types=['bev', '3d']):
        """AICV evaluation.

        Args:
            gt_annos (list[dict]): Contain gt information of each sample.
            dt_annos (list[dict]): Contain detected information of each sample.
            current_classes (list[str]): Classes to evaluation.
            eval_types (list[str], optional): Types to eval.
                Defaults to ['bev', '3d'].

        Returns:
            tuple: String and dict of evaluation results.
        """
        assert len(eval_types) > 0, 'must contain at least one evaluation type'
        if 'aos' in eval_types:
            assert 'bbox' in eval_types, 'must evaluate bbox when evaluating aos'
        overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5],
                                [0.7, 0.5, 0.5, 0.7, 0.5],
                                [0.7, 0.5, 0.5, 0.7, 0.5]])
        overlap_0_5 = np.array([[0.5, 0.25, 0.25, 0.5, 0.25],
                                [0.5, 0.25, 0.25, 0.5, 0.25],
                                [0.5, 0.25, 0.25, 0.5, 0.25]])
        # 2 overlaps; 3 metrics: [bbox, bev, 3d]; 5 classes
        min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 3, 5]
        class_to_name = {
            0: 'Car',
            1: 'Pedestrian',
            2: 'Cyclist',
            3: 'Van',
            4: 'Person_sitting',
        }
        name_to_class = {v: n for n, v in class_to_name.items()}
        if not isinstance(current_classes, (list, tuple)):
            current_classes = [current_classes]
        current_classes_int = []
        for curcls in current_classes:
            if isinstance(curcls, str):
                current_classes_int.append(name_to_class[curcls])
            else:
                current_classes_int.append(curcls)
        current_classes = current_classes_int
        min_overlaps = min_overlaps[:, :, current_classes]
        result = ''
        mAP40_bev, mAP40_3d = self.do_eval(gt_annos, dt_annos, current_classes, 
                                           min_overlaps, eval_types)

        ret_dict = {}
        difficulty = ['moderate']
        # Calculate AP40
        result += '\n----------- AP40 Results ------------\n'
        mean_bev = mAP40_bev.mean(axis=0)
        mean_3d = mAP40_3d.mean(axis=0)

        # overlap 0.7
        result += '\t'
        for j, curcls in enumerate(current_classes):
            result += '{}({}) '.format(class_to_name[curcls], min_overlaps[0, 1, j])
        result += 'mAP'
        for i in range(min_overlaps.shape[0]):
            result += '\n{}\t'.format(eval_types[i])
            for j, curcls in enumerate(current_classes):
                if eval_types[i] == 'bev':
                    result += '{:.3f}\t'.format(*mAP40_bev[j, :, 0])
                elif eval_types[i] == '3d':
                    result += '{:.3f}\t'.format(*mAP40_3d[j, :, 0])
            overall = mean_bev if eval_types[i] == 'bev' else mean_3d
            result += '{:.2f}'.format(*overall[:, 0])
        # overlap 0.5
        result += '\n\n\t'
        for j, curcls in enumerate(current_classes):
            result += '{}({}) '.format(class_to_name[curcls], min_overlaps[1, 1, j])
        result += 'mAP'
        for i in range(min_overlaps.shape[0]):
            result += '\n{}\t'.format(eval_types[i])
            for j, curcls in enumerate(current_classes):
                if eval_types[i] == 'bev':
                    result += '{:.3f}\t'.format(*mAP40_bev[j, :, 1])
                elif eval_types[i] == '3d':
                    result += '{:.3f}\t'.format(*mAP40_3d[j, :, 1])
            overall = mean_bev if eval_types[i] == 'bev' else mean_3d
            result += '{:.2f}'.format(*overall[:, 1])

        return result, ret_dict
    

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in AICV protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str, optional): The prefix of pkl files, including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str, optional): The prefix of submission data.
                If not specified, the submission data will not be generated.
                Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, pklfile_prefix)
        gt_annos = self.format_gt_annos()

        if isinstance(result_files, dict):
            ap_dict = dict()
            for name, result_files_ in result_files.items():
                eval_types = ['bev', '3d']
                if 'img' in name:
                    eval_types = ['bbox']
                ap_result_str, ap_dict_ = self.aicv_eval(
                    gt_annos,
                    result_files_,
                    self.CLASSES,
                    eval_types=eval_types)
                for ap_type, ap in ap_dict_.items():
                    ap_dict[f'{name}/{ap_type}'] = float('{:.4f}'.format(ap))

                print_log(
                    f'Results of {name}:\n' + ap_result_str, logger=logger)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return ap_dict

    def bbox2result_kitti(self,
                          net_outputs,
                          class_names,
                          pklfile_prefix=None,
                          submission_prefix=None):
        """Convert 3D detection results to kitti format for evaluation and test
        submission.

        Args:
            net_outputs (list[np.ndarray]): List of array storing the
                inferenced bounding boxes and scores.
            class_names (list[String]): A list of class names.
            pklfile_prefix (str): The prefix of pkl file.
            submission_prefix (str): The prefix of submission file.

        Returns:
            list[dict]: A list of dictionaries with the kitti format.
        """
        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        det_annos = []
        print('\nConverting prediction to AICV format')
        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            annos = []
            info = self.data_infos[idx]
            sample_idx = info['image']['image_idx']
            box_dict = self.convert_valid_bboxes(pred_dicts, info)
            anno = {
                'name': [],
                'truncated': [],
                'occluded': [],
                'alpha': [],
                'bbox': [],
                'dimensions': [],
                'location': [],
                'rotation_y': [],
                'score': []
            }
            if len(box_dict['box3d_lidar']) > 0:
                scores = box_dict['scores']
                box_preds_lidar = box_dict['box3d_lidar']
                label_preds = box_dict['label_preds']

                for box_lidar, score, label in zip(
                        box_preds_lidar, scores, label_preds):
                    anno['name'].append(class_names[int(label)])
                    anno['truncated'].append(0.0)
                    anno['occluded'].append(0)
                    anno['alpha'].append(-10)
                    anno['bbox'].append(np.array([0, 0, 100, 100]))

                    x = box_lidar[0]
                    y = box_lidar[1]
                    z = box_lidar[2]
                    l = box_lidar[3]
                    w = box_lidar[4]
                    h = box_lidar[5]
                    r = box_lidar[6]
                    
                    r = -r - np.pi / 2
                    if r < -np.pi:
                        r = r + 2 * np.pi
                    anno['dimensions'].append(np.array([l, h, w]))
                    anno['location'].append(np.array([-y, -z, x]))
                    anno['rotation_y'].append(r)
                    anno['score'].append(score)

                anno = {k: np.stack(v) for k, v in anno.items()}
                annos.append(anno)
            else:
                anno = {
                    'name': np.array([]),
                    'truncated': np.array([]),
                    'occluded': np.array([]),
                    'alpha': np.array([]),
                    'bbox': np.zeros([0, 4]),
                    'dimensions': np.zeros([0, 3]),
                    'location': np.zeros([0, 3]),
                    'rotation_y': np.array([]),
                    'score': np.array([]),
                }
                annos.append(anno)

            if submission_prefix is not None:
                curr_file = f'{submission_prefix}/{sample_idx:06d}.txt'
                with open(curr_file, 'w') as f:
                    bbox = anno['bbox']
                    loc = anno['location']
                    dims = anno['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print(
                            '{} -1 -1 {:.4f} {:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} '
                            '{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}'.format(
                                anno['name'][idx], anno['alpha'][idx],
                                bbox[idx][0], bbox[idx][1], bbox[idx][2],
                                bbox[idx][3], dims[idx][1], dims[idx][2],
                                dims[idx][0], loc[idx][0], loc[idx][1],
                                loc[idx][2], anno['rotation_y'][idx],
                                anno['score'][idx]),
                            file=f)

            annos[-1]['sample_idx'] = np.array(
                [sample_idx] * len(annos[-1]['score']), dtype=np.int64)

            det_annos += annos

        if pklfile_prefix is not None:
            if not pklfile_prefix.endswith(('.pkl', '.pickle')):
                out = f'{pklfile_prefix}.pkl'
            mmcv.dump(det_annos, out)
            print(f'Result is saved to {out}.')

        return det_annos

    def convert_valid_bboxes(self, box_dict, info):
        """Convert the predicted boxes into valid ones.

        Args:
            box_dict (dict): Box dictionaries to be converted.

                - boxes_3d (:obj:`LiDARInstance3DBoxes`): 3D bounding boxes.
                - scores_3d (torch.Tensor): Scores of boxes.
                - labels_3d (torch.Tensor): Class labels of boxes.
            info (dict): Data info.

        Returns:
            dict: Valid predicted boxes.

                - bbox (np.ndarray): 2D bounding boxes.
                - box3d_camera (np.ndarray): 3D bounding boxes in
                    camera coordinate.
                - box3d_lidar (np.ndarray): 3D bounding boxes in
                    LiDAR coordinate.
                - scores (np.ndarray): Scores of boxes.
                - label_preds (np.ndarray): Class label predictions.
                - sample_idx (int): Sample index.
        """
        # TODO: refactor this function
        box_preds = box_dict['boxes_3d']
        scores = box_dict['scores_3d']
        labels = box_dict['labels_3d']
        sample_idx = info['image']['image_idx']
        box_preds.limit_yaw(offset=0.5, period=np.pi * 2)

        if len(box_preds) == 0:
            return dict(
                bbox=np.zeros([0, 4]),
                box3d_camera=np.zeros([0, 7]),
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

        # check box_preds
        limit_range = box_preds.tensor.new_tensor(self.pcd_limit_range)
        valid_pcd_inds = ((box_preds.center > limit_range[:3]) &
                          (box_preds.center < limit_range[3:]))
        valid_inds = valid_pcd_inds.all(-1)

        if valid_inds.sum() > 0:
            return dict(
                box3d_lidar=box_preds[valid_inds].tensor.numpy(),
                scores=scores[valid_inds].numpy(),
                label_preds=labels[valid_inds].numpy(),
                sample_idx=sample_idx)
        else:
            return dict(
                box3d_lidar=np.zeros([0, 7]),
                scores=np.zeros([0]),
                label_preds=np.zeros([0, 4]),
                sample_idx=sample_idx)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        if self.modality['use_camera']:
            pipeline.insert(0, dict(type='LoadImageFromFile'))
        return Compose(pipeline)

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['point_cloud']['velodyne_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points, img_metas, img = self._extract_data(
                i, pipeline, ['points', 'img_metas', 'img'])
            points = points.numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

            # multi-modality visualization
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # need to transpose channel to first dim
                img = img.transpose(1, 2, 0)
                show_pred_bboxes = LiDARInstance3DBoxes(
                    pred_bboxes, origin=(0.5, 0.5, 0))
                show_gt_bboxes = LiDARInstance3DBoxes(
                    gt_bboxes, origin=(0.5, 0.5, 0))
                show_multi_modality_result(
                    img,
                    show_gt_bboxes,
                    show_pred_bboxes,
                    img_metas['lidar2img'],
                    out_dir,
                    file_name,
                    box_mode='lidar',
                    show=show)
