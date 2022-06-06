# Copyright (c) OpenMMLab. All rights reserved.
import gc
import io as sysio

import numba
import numpy as np

from mmdet3d.core.evaluation.kitti_utils.eval import eval_class, get_mAP40

def do_eval(gt_annos,
            dt_annos,
            current_classes,
            min_overlaps,
            eval_types=['bev', '3d']):
    # min_overlaps: [num_minoverlap, metric, num_class]
    difficultys = [1]
    mAP40_bev = None
    if 'bev' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0,
                         min_overlaps)
        mAP40_bev = get_mAP40(ret['precision'])

    mAP40_3d = None
    if '3d' in eval_types:
        ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1,
                         min_overlaps)
        mAP40_3d = get_mAP40(ret['precision'])
    return (mAP40_bev, mAP40_3d)

def aicv_eval(gt_annos,
              dt_annos,
              current_classes,
              eval_types=['bev', '3d']):
    """KITTI evaluation.

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
                            [0.7, 0.5, 0.5, 0.7, 0.5]])
    overlap_0_5 = np.array([[0.5, 0.25, 0.25, 0.5, 0.25],
                            [0.5, 0.25, 0.25, 0.5, 0.25]])
    # 2 overlaps; 2 metrics: [bev, 3d]; 5 classes
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)  # [2, 2, 5]
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
    # check whether alpha is valid
    compute_aos = False
    pred_alpha = False
    valid_alpha_gt = False
    for anno in dt_annos:
        mask = (anno['alpha'] != -10)
        if anno['alpha'][mask].shape[0] != 0:
            pred_alpha = True
            break
    for anno in gt_annos:
        if anno['alpha'].size != 0 and anno['alpha'][0] != -10:
            valid_alpha_gt = True
            break
    compute_aos = (pred_alpha and valid_alpha_gt)
    if compute_aos:
        eval_types.append('aos')

    mAP40_bev, mAP40_3d = do_eval(gt_annos, dt_annos, current_classes, 
                                  min_overlaps, eval_types)

    ret_dict = {}
    difficulty = ['moderate']

    # Calculate AP40
    result += '\n----------- AP40 Results ------------\n\n'
    for j, curcls in enumerate(current_classes):
        # mAP threshold array: [num_minoverlap, metric, class]
        # mAP result: [num_class, num_diff, num_minoverlap]
        curcls_name = class_to_name[curcls]
        for i in range(min_overlaps.shape[0]):
            # prepare results for print
            result += ('{} AP40@{:.2f}, {:.2f}:\n'.format(
                curcls_name, *min_overlaps[i, :, j]))
            if mAP40_bev is not None:
                result += 'bev  AP40: {:.4f}\n'.format(*mAP40_bev[j, :, i])
            if mAP40_3d is not None:
                result += '3d   AP40: {:.4f}\n'.format(*mAP40_3d[j, :, i])

            # prepare results for logger
            for idx in range(1):
                if i == 0:
                    postfix = f'{difficulty[idx]}_strict'
                else:
                    postfix = f'{difficulty[idx]}_loose'
                prefix = f'AICV/{curcls_name}'
                if mAP40_3d is not None:
                    ret_dict[f'{prefix}_3D_AP40_{postfix}'] = mAP40_3d[j, idx, i]
                if mAP40_bev is not None:
                    ret_dict[f'{prefix}_BEV_AP40_{postfix}'] = mAP40_bev[j, idx, i]

    # calculate mAP40 over all classes if there are multiple classes
    if len(current_classes) > 1:
        # prepare results for print
        result += ('\nOverall AP40@{}:\n'.format(*difficulty))
        if mAP40_bev is not None:
            mAP40_bev = mAP40_bev.mean(axis=0)
            result += 'bev  AP40: {:.4f}\n'.format(*mAP40_bev[:, 0])
        if mAP40_3d is not None:
            mAP40_3d = mAP40_3d.mean(axis=0)
            result += '3d   AP40: {:.4f}\n'.format(*mAP40_3d[:, 0])

        # prepare results for logger
        for idx in range(1):
            postfix = f'{difficulty[idx]}'
            if mAP40_3d is not None:
                ret_dict[f'AICV/Overall_3D_AP40_{postfix}'] = mAP40_3d[idx, 0]
            if mAP40_bev is not None:
                ret_dict[f'AICV/Overall_BEV_AP40_{postfix}'] = mAP40_bev[idx, 0]

    return result, ret_dict
