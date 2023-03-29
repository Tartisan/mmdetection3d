import os
import os.path as osp
import time
import numpy as np
from mmdet3d.core.visualizer.open3d_vis import Visualizer
from tools.visualizer import pypcd

VAL_LIST = None
LABEL_DIR = None

# PCD_DIR = '/media/Data/caros/record/BUS2015_default_47_20220830113505_20220830113605-144372/000022.bin'
PCD_DIR = '/media/Data/datasets/at128/kitti_format/training/velodyne/000199.bin'
RESULT_DIR = '/media/Data/datasets/at128/20221210-0.32_84.48m_dw2/result_output/000199.txt'
# VAL_LIST = '../../ImageSets/val.txt'
LABEL_DIR = '/media/Data/datasets/at128/kitti_format/training/label/000199.txt'
SUFFIX='.bin'
LOAD_DIM = 5

# https://www.rapidtables.com/web/color/RGB_Color.html
PALETTE = [[30, 144, 255],  # dodger blue
           [0, 255, 255],   # 青色
           [255, 215, 0],   # 金黄色
           [160, 32, 240],  # 紫色
           [3, 168, 158],   # 锰蓝
           [255, 0, 0],     # 红色
           [0, 0, 0],       # 黑色
           [255, 97, 0],    # 橙色
           [0, 201, 87]]    # 翠绿色

ObjectType2Lable = {'smallMot': 0, 'pedestrian': 1, 'nonMot': 2, 'trafficcone': 3, 'unknown': 4}

def show_result_meshlab(vis,
                        data,
                        result,
                        out_dir=None,
                        gt_bboxes=None,
                        score_thr=0.0,
                        snapshot=False):
    """Show 3D detection result by meshlab."""
    points = data
    pred_bboxes = result[:, 0:7]
    pred_labels = result[:, 7]

    vis.o3d_visualizer.clear_geometries()
    vis.add_points(points)
    if gt_bboxes is not None:
        vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 1, 0), 
                       points_in_box_color=(0.5, 0.5, 0.5))
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
                palette = [c / 255.0 for c in PALETTE[i]]
                vis.add_bboxes(
                    bbox3d=np.array(labelDict[i]),
                    bbox_color=palette, 
                    points_in_box_color=palette)

    ctr = vis.o3d_visualizer.get_view_control()
    ctr.set_lookat([0,0,0])
    ctr.set_front([-1,-1,1])    # 设置垂直指向屏幕外的向量
    ctr.set_up([0,0,1])         # 设置指向屏幕上方的向量
    ctr.set_zoom(0.1)

    vis.o3d_visualizer.poll_events()
    vis.o3d_visualizer.update_renderer()


def load_cybertron_perception_results(result_path):
    results = []
    with open(result_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            if line == '' or line == ' ' or line == '\n':
                continue
            label = ObjectType2Lable[line.split(' ')[5]]
            # if label > 2:
            #     continue
            result = [float(x) for x in line.split(' ')[6:13]]
            result.append(label)
            results.append(result)
    return np.array(results)


def load_label(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    content = [line.strip().split(' ') for line in lines]
    dimensions = np.array([[float(info) for info in x[8:11]]
                           for x in content
                           ]).reshape(-1, 3)[:, [2, 0, 1]]
    location = np.array([[float(info) for info in x[11:14]]
                         for x in content]).reshape(-1, 3)
    rotation_y = np.array([float(x[14]) for x in content]).reshape(-1)
    gt_bboxes = np.concatenate((location, dimensions, rotation_y.reshape(-1, 1)), axis=1)
    return gt_bboxes


def dataloader(pcd_path, result_path):
    if SUFFIX == '.bin':
        cloud = np.fromfile(
            pcd_path, dtype=np.float32, count=-1).reshape([-1, LOAD_DIM])
    elif SUFFIX == '.pcd':
        pcd = pypcd.PointCloud.from_path(pcd_path)
        cloud = np.stack([pcd.pc_data['x'], 
                          pcd.pc_data['y'], 
                          pcd.pc_data['z']]).transpose(1, 0)

    results = load_cybertron_perception_results(result_path)
    return cloud , results


def main():
    # pointcloud
    pcd_list = []
    if osp.isdir(PCD_DIR):
        if VAL_LIST is None:
            for path, dir_list, file_list in os.walk(osp.abspath(PCD_DIR)):
                for file_name in file_list:
                    pcd_file = osp.join(path, file_name)
                    if SUFFIX in pcd_file:
                        pcd_list.append(pcd_file)
        else:
            with open(osp.join(PCD_DIR, VAL_LIST)) as f:
                lines = f.readlines()
            for line in lines:
                pcd_file = osp.join(PCD_DIR, line.split('\n')[0].zfill(6) + SUFFIX)
                pcd_list.append(pcd_file)
    elif osp.isfile(PCD_DIR):
        pcd_list.append(PCD_DIR)
    pcd_list.sort()

    # detection result
    result_list = []
    if osp.isdir(RESULT_DIR):
        for path, dir_list, file_list in os.walk(osp.abspath(RESULT_DIR)):
            for file_name in file_list:
                label_file = osp.join(path, file_name)
                result_list.append(label_file)
    elif osp.isfile(RESULT_DIR):
        result_list.append(RESULT_DIR)
    result_list.sort()

    # label
    label_list = []
    if LABEL_DIR is not None:
        if osp.isdir(LABEL_DIR):
            if VAL_LIST is None:
                for path, dir_list, file_list in os.walk(osp.abspath(LABEL_DIR)):
                    for file_name in file_list:
                        label_file = osp.join(path, file_name)
                        label_list.append(label_file)
            else:
                with open(osp.join(PCD_DIR, VAL_LIST)) as f:
                    lines = f.readlines()
                for line in lines:
                    label_file = osp.join(LABEL_DIR, line.split('\n')[0].zfill(6) + '.txt')
                    label_list.append(label_file)
        elif osp.isfile(LABEL_DIR):
            label_list.append(LABEL_DIR)
    label_list.sort()

    # assert(len(pcd_list) == len(result_list))

    # init visualizer
    vis = Visualizer(None)
    for i in range(len(pcd_list)):
        print(pcd_list[i])
        # test a single image
        cloud, results = dataloader(pcd_list[i], result_list[i])
        gt_bboxes = None
        if label_list != []:
            assert(len(pcd_list) == len(label_list))
            gt_bboxes = load_label(label_list[i])
        # show the results
        show_result_meshlab(
            vis, 
            cloud,
            results,
            None,
            gt_bboxes, 
            score_thr=0.0,
            snapshot=False)
        time.sleep(0.3)
    vis.show()


if __name__ == "__main__":
    main()