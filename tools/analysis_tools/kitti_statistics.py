import os
import os.path as osp
import mmcv
import math
import argparse
import json
from attr import s
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn_fig2grid as sfg


KITTI_CLASSES = ['Car', 'Cyclist', 'Pedestrian']

pd_all_classes = {c: [] for c in KITTI_CLASSES}
pd_total = []

parser = argparse.ArgumentParser()
parser.add_argument('--root-path', type=str, default='data/kitti',
                    help='path of the dataset')
args = parser.parse_args()

trainval_infos = mmcv.load(osp.join(args.root_path, 'kitti_infos_trainval.pkl'))
sample_num = 0
for infos in trainval_infos:
    for i in range(len(infos['annos']['name'])):
        type = infos['annos']['name'][i]
        if type in KITTI_CLASSES:
            l = infos['annos']['dimensions'][i, 0]
            h = infos['annos']['dimensions'][i, 1]
            w = infos['annos']['dimensions'][i, 2]
            bh = infos['annos']['location'][i, 1]
            pd_line = [sample_num, type, l, w, h, bh]
            pd_all_classes[type].append(pd_line)
            pd_total.append(pd_line)
    sample_num += 1

# print sample statistics
print('samples num: ', sample_num)
for k, v in pd_all_classes.items():
    print('load {} {},\taverage {:.2f} per sample'.format(len(v), k, len(v)/sample_num))
print('\n')

# anchor
sns.set()
total = pd.DataFrame(pd_total, columns=['index', 'class', 'l', 'w', 'h', 'bottom_h'])
sns.jointplot(x="l", y="w", data=total, kind='kde', hue='class')
plt.show()

fig_hist = plt.figure(figsize=(18, 10))
fig_kde_lw = plt.figure(figsize=(18, 10))
fig_kde_h = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, math.ceil(len(KITTI_CLASSES)/2))
count = 0
for type, pd_single_class in pd_all_classes.items():
    data = pd.DataFrame(pd_single_class, columns=[
                        'index', 'class', 'l', 'w', 'h', 'bottom_h'])
    print(type, 'dimension statistics:\n', data.describe(), '\n')

    jg0 = sns.jointplot(x="l", y="w", data=data, marker="+")
    sfg.SeabornFig2Grid(jg0, fig_hist, gs[count])
    gs.tight_layout(fig_hist)

    jg1 = sns.jointplot(x="l", y="w", data=data, kind='kde', fill=True)
    sfg.SeabornFig2Grid(jg1, fig_kde_lw, gs[count])
    gs.tight_layout(fig_kde_lw)

    jg2 = sns.jointplot(x="h", y="bottom_h", data=data, kind='kde', fill=True)
    sfg.SeabornFig2Grid(jg2, fig_kde_h, gs[count])
    gs.tight_layout(fig_kde_h)
    count += 1
plt.show()

