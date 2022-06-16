# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import numpy as np
import os
import os.path as osp

import open3d as o3d
import mmcv
from mmdet3d.apis import inference_detector, init_model
from mmdet3d.core.visualizer.open3d_vis import Visualizer, add_pts_boxes
from tools.data_converter.kitti_data_utils import get_label_anno

# http://www1.ynao.ac.cn/~jinhuahe/know_base/othertopics/computerissues/RGB_colortable.htm
PALETTE = [[0, 255, 0],     # 绿色
           [0, 255, 255],   # 青色
           [255, 153, 18],  # 镉黄
           [255, 0, 255],   # 深红
           [3, 138, 158],   # 锰蓝
           [160, 32, 240],  # 紫色
           [255, 255, 255]] # 黑色


def show_result_meshlab(vis, 
                        data,
                        result,
                        out_dir=None,
                        gt_bboxes=None, 
                        score_thr=0.0,
                        snapshot=False):
    """Show 3D detection result by meshlab."""
    points = data['points'][0][0].cpu().numpy()
    pts_filename = data['img_metas'][0][0]['pts_filename']
    file_name = osp.split(pts_filename)[-1].split('.')[0]

    if 'pts_bbox' in result[0].keys():
        pred_bboxes = result[0]['pts_bbox']['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['pts_bbox']['scores_3d'].numpy()
        pred_labels = result[0]['pts_bbox']['labels_3d'].numpy()
    else:
        pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
        pred_scores = result[0]['scores_3d'].numpy()
        pred_labels = result[0]['labels_3d'].numpy()

    # filter out low score bboxes for visualization
    if score_thr > 0:
        inds = pred_scores > score_thr
        pred_bboxes = pred_bboxes[inds]
        pred_labels = pred_labels[inds]

    vis.o3d_visualizer.clear_geometries()
    vis.add_points(points)
    if pred_bboxes is not None:
        if pred_labels is None:
            vis.add_bboxes(bbox3d=pred_bboxes)
        else:
            labelDict = {}
            for j in range(len(pred_labels)):
                i = int(pred_labels[j])
                if labelDict.get(i) is None:
                    labelDict[i] = []
                labelDict[i].append(pred_bboxes[j])
            for i in labelDict:
                vis.add_bboxes(
                    bbox3d=np.array(labelDict[i]),
                    bbox_color=([c / 255.0 for c in PALETTE[i]]))
    if gt_bboxes is not None:
        vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))

    ctr = vis.o3d_visualizer.get_view_control()
    ctr.set_lookat([0,0,0])
    ctr.set_front([-1,-1,1])    # 设置垂直指向屏幕外的向量
    ctr.set_up([0,0,1])         # 设置指向屏幕上方的向量
    ctr.set_zoom(0.1)

    vis.o3d_visualizer.poll_events()
    vis.o3d_visualizer.update_renderer()

    if out_dir is not None and snapshot:
        result_path = osp.join(out_dir, file_name)
        mmcv.mkdir_or_exist(result_path)
        save_path = osp.join(result_path, 
                             f'{file_name}_online.png') if snapshot else None
        vis.o3d_visualizer.capture_screen_image(save_path)


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
        '--out-dir', default=None, help='dir to save results')
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
    
    pcd_list = []
    if osp.isdir(args.pcd):
        for path, dir_list, file_list in os.walk(osp.abspath(args.pcd)):
            for file_name in file_list:
                pcd_file = osp.join(path, file_name)
                if ".bin" in pcd_file:
                    pcd_list.append(pcd_file)
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
            vis, 
            data,
            result,
            args.out_dir,
            gt_bboxes, 
            score_thr=args.score_thr,
            snapshot=args.snapshot)
    
    vis.show()


if __name__ == '__main__':
    main()
