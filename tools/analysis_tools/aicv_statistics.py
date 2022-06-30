import os
import os.path as osp
import argparse
import json

AICV_CLASSES = [
    'smallMot', 'bigMot', 'OnlyTricycle', 'OnlyBicycle', 
    'Tricyclist', 'bicyclist', 'motorcyclist', 'pedestrian', 
    'TrafficCone', 'others', 'fog', 'stopBar', 'smallMovable', 
    'smallUnmovable', 'crashBarrel', 'safetyBarrier', 'sign'
]

CUSTOM_CLASSES = [
    'Car', 'Cyclist', 'Pedestrian', 'NonMot', 'TrafficCone', 'Others'
]

aicv_to_custom_class_map = {
    'smallMot': 'Car',
    'bigMot': 'Car',
    'OnlyTricycle': 'NonMot',
    'OnlyBicycle': 'NonMot',
    'Tricyclist': 'Cyclist',
    'bicyclist': 'Cyclist',
    'motorcyclist': 'Cyclist',
    'pedestrian': 'Pedestrian',
    'TrafficCone': 'TrafficCone', 
    'stopBar': 'Others', 
    'crashBarrel': 'Others', 
    'safetyBarrier': 'Others', 
    'sign': 'Others', 
    'smallMovable': 'Others',
    'smallUnmovable': 'Others', 
    'fog': 'Others',
    'others': 'Others'
}

# [counter, l, w, h, bottom_height]
sample_counter = {c: [0, 0.0, 0.0, 0.0, 0.0] for c in CUSTOM_CLASSES}

parser = argparse.ArgumentParser()
parser.add_argument('root_path', metavar='data/hesai40', help='path of the dataset')

args = parser.parse_args()

sample_num = 0
with open(osp.join(args.root_path, 'result.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines[1:]:
        infos = json.loads(line.split('\t')[1])
        if 'labelData' not in infos.keys():
            continue
        sample_num += 1
        annotations = infos['labelData']['result']
        for anno in annotations:
            type = anno['type']
            z = anno['position']['z']
            l = anno['size'][0]
            w = anno['size'][1]
            h = anno['size'][2]
            if type in AICV_CLASSES:
                sample_counter[aicv_to_custom_class_map[type]][0] += 1
                sample_counter[aicv_to_custom_class_map[type]][1] += l
                sample_counter[aicv_to_custom_class_map[type]][2] += w
                sample_counter[aicv_to_custom_class_map[type]][3] += h
                sample_counter[aicv_to_custom_class_map[type]][4] += (z - h/2)
            else:
                print('new class: ', type)

print('samples num: ', sample_num)
for k, v in sample_counter.items():
    print('load {} {}, average dimension: [{:.2f}, {:.2f}, {:.2f}], bottom_height: {:.2f}'.format(
        v[0], k, v[1]/(v[0]+1e-4), v[2]/(v[0]+1e-4), v[3]/(v[0]+1e-4), v[4]/(v[0]+1e-4)))
print('\neach sample contains:')
for k, v in sample_counter.items():
    print('{:.2f} {}'.format(v[0]/sample_num, k))