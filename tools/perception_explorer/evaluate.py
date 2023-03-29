import os
import os.path as osp
import numpy as np
import torch

from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets.aicv_dataset import AicvDataset

result_path = '/media/Data/caros/baidu/robobus/perception-explorer/output/result_output'
data_root = '/media/Data/datasets/hesai40-bp/kitti_format'
ann_file = data_root + '/hesai40-bp_infos_val.pkl'


ObjectType2Lable = {'smallMot': 0, 'nonMot': 1, 'pedestrian': 2}

def load_cybertron_perception_results(data_root, result_path):
    from mmdet3d.core.bbox import LiDARInstance3DBoxes
    result_list = []
    with open(osp.join(data_root, 'ImageSets/val.txt')) as f:
        result_list = f.readlines()
    results = []
    for file_path in result_list:
        result_dict = {'pts_bbox': {'boxes_3d': None, 
                                    'scores_3d': None, 
                                    'labels_3d': None}}
        bboxes = []
        scores = []
        labels = []
        file = file_path.split('\n')[0].zfill(6) + '.txt'
        with open(osp.join(result_path, file), 'r') as f:
            lines = f.readlines()
            for line in lines[1:]:
                if line == '' or line == ' ' or line == '\n':
                    continue
                values = line.split(' ')
                if values[5] not in ObjectType2Lable.keys():
                    continue
                label = ObjectType2Lable[values[5]]
                scores.append(float(values[4]))
                labels.append(label)
                bbox = [float(x) for x in values[6:13]]
                bboxes.append(bbox)

        result_dict['pts_bbox']['boxes_3d'] = LiDARInstance3DBoxes(
                torch.tensor(bboxes), origin=(0.5, 0.5, 0))
        result_dict['pts_bbox']['scores_3d'] = torch.tensor(scores)
        result_dict['pts_bbox']['labels_3d'] = torch.tensor(labels)
        results.append(result_dict)
    
    return results


def main():
    CLASSES = ('SmallMot', 'Cyclist', 'Pedestrian')
    MODALITY = {'use_lidar': True, 'use_camera': False}
    aicv = AicvDataset(data_root, 
                       ann_file, 
                       'training', 
                       classes=CLASSES, 
                       modality=MODALITY)
    with torch.no_grad():
        results = load_cybertron_perception_results(data_root, result_path)
        print(aicv.evaluate(results, metric='mAP'))


if __name__ == "__main__":
    main()