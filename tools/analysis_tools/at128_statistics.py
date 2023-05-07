import os
import os.path as osp
import pickle
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn_fig2grid as sfg


CUSTOM_CLASSES = ['Car', 'Pedestrian', 'Bicycle', 'TrafficCone']


if __name__ == '__main__':
    f = open('data/at128/kitti_format/at128_infos_val.pkl', 'rb')
    pkl_infos = pickle.load(f)

    data = []
    data_all_class = {c: [] for c in CUSTOM_CLASSES}
    for info in pkl_infos:
        classes = info['annos']['name']
        num_points_in_gt = info['annos']['num_points_in_gt']
        location = info['annos']['location']
        assert(classes.size == num_points_in_gt.size)
        for i in range(classes.size):
            if classes[i] == 'Car' and  num_points_in_gt[i] == 0:
                print(info['point_cloud']['velodyne_path'])
            line = [classes[i], num_points_in_gt[i], location[i][0], location[i][1]]
            data.append(line)
            data_all_class[classes[i]].append(line)
    
    for c, data_single_class in data_all_class.items():
        pd_infos = pd.DataFrame(data_single_class, columns=[
                                'class', 'num_points_in_gt', 'x', 'y'])
        print(c, 'dimension statistics:\n', pd_infos.describe(), '\n')
