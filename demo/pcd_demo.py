# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import numpy as np

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
        '--out-dir', type=str, default='demo', help='dir to save results')
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
    # test a single image
    result, data = inference_detector(model, args.pcd)
    gt_bboxes = None
    if args.label is not None:
        anno = get_label_anno(args.label)
        gt_bboxes = np.concatenate((anno['location'], 
                                    anno['dimensions'], 
                                    anno['rotation_y'].reshape(-1, 1)), axis=1)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        gt_bboxes, 
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='det')


if __name__ == '__main__':
    main()
