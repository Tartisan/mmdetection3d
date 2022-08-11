# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import mmcv
from mmcv import Config

from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    parser.add_argument(
        '--non-blocking', action='store_true', 
        help='whether to do non-blocking visualization')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # build the dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)

    show_dir = args.show_dir
    if show_dir is None:
        show_dir = "tmp/" + cfg.data_root.split('/')[1]

    if getattr(dataset, 'show', None) is not None:
        # data loading pipeline for showing
        eval_pipeline = cfg.get('eval_pipeline', {})
        if eval_pipeline:
            dataset.show(results, show_dir, non_blocking=args.non_blocking, 
                         pipeline=eval_pipeline)
        else:
            dataset.show(results, show_dir)  # use default pipeline
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()
