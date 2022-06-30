# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import numpy as np
import os
import os.path as osp

from mmdet3d.apis import inference_detector, init_model, show_result_meshlab
from tools.data_converter.kitti_data_utils import get_label_anno


def main():
    parser = ArgumentParser()
    parser.add_argument('pcd', help='Point cloud file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--label', help='groundtruth label file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.0, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', default='tmp', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    
    non_blocking = False
    pcd_list = []
    if osp.isdir(args.pcd):
        for path, dir_list, file_list in os.walk(osp.abspath(args.pcd)):
            for file_name in file_list:
                pcd_file = osp.join(path, file_name)
                if ".bin" in pcd_file:
                    pcd_list.append(pcd_file)
        non_blocking = True
    elif osp.isfile(args.pcd):
        pcd_list.append(args.pcd)
    pcd_list.sort()

    label_list = []
    if args.label is not None:
        if osp.isdir(args.label):
            for path, dir_list, file_list in os.walk(osp.abspath(args.label)):
                for file_name in file_list:
                    label_file = osp.join(path, file_name)
                    label_list.append(label_file)
        elif osp.isfile(args.label):
            label_list.append(args.label)

    # init visualizer
    vis = None
    if non_blocking:
        from mmdet3d.core.visualizer.open3d_vis import Visualizer
        vis = Visualizer(None)

    for i in range(len(pcd_list)):
        # test a single image
        result, data = inference_detector(model, pcd_list[i])
        gt_bboxes = None
        if label_list != []:
            anno = get_label_anno(label_list[i])
            gt_bboxes = np.concatenate((anno['location'], 
                                        anno['dimensions'], 
                                        anno['rotation_y'].reshape(-1, 1)), axis=1)
        # show the results
        show_result_meshlab(
            data,
            result,
            args.out_dir,
            gt_bboxes, 
            score_thr=args.score_thr,
            show=args.show,
            snapshot=args.snapshot,
            task='det',
            visualizer=vis)


if __name__ == '__main__':
    main()
