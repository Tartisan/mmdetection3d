import os
import os.path as osp
import math
import argparse
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn_fig2grid as sfg

AICV_CLASSES = [
    'smallMot', 'bigMot', 'OnlyTricycle', 'OnlyBicycle',
    'Tricyclist', 'bicyclist', 'motorcyclist', 'pedestrian',
    'TrafficCone', 'others', 'fog', 'stopBar', 'smallMovable',
    'smallUnmovable', 'crashBarrel', 'safetyBarrier', 'sign'
]

# CUSTOM_CLASSES = [
#     'Car', 'Cyclist', 'Pedestrian', 'TrafficCone', 'NonMot', 'Barrier'
# ]
# CUSTOM_CLASSES = ['Car', 'Pedestrian', 'Bicycle', 'TrafficCone', 'Barrier']
CUSTOM_CLASSES = ['Car', 'Pedestrian', 'Bicycle', 'TrafficCone']

aicv_to_custom = {
    'smallMot': 'Car',
    'bigMot': 'Car',
    'OnlyTricycle': 'Bicycle',
    'OnlyBicycle': 'Bicycle',
    'Tricyclist': 'Bicycle',
    'bicyclist': 'Bicycle',
    'motorcyclist': 'Bicycle',
    'pedestrian': 'Pedestrian',
    'TrafficCone': 'TrafficCone',
    'stopBar': 'DontCare',
    'crashBarrel': 'DontCare',
    'safetyBarrier': 'DontCare',
    'sign': 'DontCare',
    'smallMovable': 'DontCare',
    'smallUnmovable': 'DontCare',
    'fog': 'DontCare',
    'others': 'DontCare'
}

point_cloud_range = [-90, -90, -5, 90, 90, 5]

pd_all_classes = {c: [] for c in CUSTOM_CLASSES}
pd_total = []

parser = argparse.ArgumentParser()
parser.add_argument('root_path', metavar='data/hesai40',
                    help='path of the dataset')
args = parser.parse_args()

frame_num = 0
with open(osp.join(args.root_path, 'result.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        infos = json.loads(line.split('\t')[1])
        # 优先使用 standardData(脑补框)
        if 'standardData' in infos.keys():
            annos = infos['standardData']
        elif 'labelData' in infos.keys():
            if 'result' in infos['labelData'].keys():
                annos = infos['labelData']['result']
            elif 'markData' in infos['labelData'].keys() and 'cube3d' in infos['labelData']['markData'].keys():
                annos = infos['labelData']['markData']['cube3d']
            else:
                continue
        else:
            continue
        for anno in annos:
            type = anno['type']
            x = anno['position']['x']
            y = anno['position']['y']
            z = anno['position']['z']
            l = anno['size'][0]
            w = anno['size'][1]
            h = anno['size'][2]
            bh = z - h/2

            if (x < point_cloud_range[0] or x > point_cloud_range[3] or
                y < point_cloud_range[1] or y > point_cloud_range[4] or
                    z < point_cloud_range[2] or z > point_cloud_range[5]):
                continue

            if type not in AICV_CLASSES:
                continue
            elif aicv_to_custom[type] in CUSTOM_CLASSES:
                # statistics
                pd_line = [frame_num, aicv_to_custom[type], l, w, h, bh]
                pd_all_classes[aicv_to_custom[type]].append(pd_line)
                pd_total.append(pd_line)

        frame_num += 1

# print sample statistics
print('frame num: ', frame_num)
total_sample_num = 0
for k, v in pd_all_classes.items():
    total_sample_num += len(v)
for k, v in pd_all_classes.items():
    print('load {} {},\taverage {:.2f} / frame, ratio {:.2f}%'.format(len(v),
          k, len(v)/frame_num, len(v)*100/total_sample_num))

# anchor
sns.set()
# total = pd.DataFrame(pd_total, columns=['index', 'class', 'l', 'w', 'h', 'bottom_h'])
# sns.jointplot(x="l", y="w", data=total, kind='kde', hue='class')
# plt.show()

# fig_hist = plt.figure(figsize=(18, 10))
# fig_kde_lw = plt.figure(figsize=(18, 10))
# fig_kde_h = plt.figure(figsize=(18, 10))
# gs = gridspec.GridSpec(2, math.ceil(len(CUSTOM_CLASSES)/2))
# count = 0
# for type, pd_single_class in pd_all_classes.items():
#     data = pd.DataFrame(pd_single_class, columns=[
#                         'index', 'class', 'l', 'w', 'h', 'bottom_h'])
#     print(type, 'dimension statistics:\n', data.describe(), '\n')

#     jg0 = sns.jointplot(x="l", y="w", data=data, marker="+")
#     sfg.SeabornFig2Grid(jg0, fig_hist, gs[count])
#     gs.tight_layout(fig_hist)

#     jg1 = sns.jointplot(x="l", y="w", data=data, kind='kde', fill=True)
#     sfg.SeabornFig2Grid(jg1, fig_kde_lw, gs[count])
#     gs.tight_layout(fig_kde_lw)

#     jg2 = sns.jointplot(x="h", y="bottom_h", data=data, kind='kde', fill=True, levels=5)
#     sfg.SeabornFig2Grid(jg2, fig_kde_h, gs[count])
#     gs.tight_layout(fig_kde_h)
#     count += 1
# plt.show()
