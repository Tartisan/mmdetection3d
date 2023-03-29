import argparse
import json
from collections import defaultdict
from turtle import width
from matplotlib.cbook import ls_mapper

import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

mpl.style.use('default')

CNNSEG_LOG = '/media/Data/caros/baidu/robobus/perception-explorer/evaluation/cnnseg_fp32/segment.log'
CPDET_LOG = '/media/Data/caros/baidu/robobus/perception-explorer/output/segment.log'


def get_fg_seg_time(file_name):
    fg_seg_time = []
    with open(file_name, 'r') as log_file:
        for line in log_file:
            if 'HybridEngineTimer' in line:
                log = line.split('HybridEngineTimer')[1].split()
                fg_seg_time.append(float(log[2]))
    return fg_seg_time


def main():
    cnnseg_time = get_fg_seg_time(CNNSEG_LOG)
    cpdet_time = get_fg_seg_time(CPDET_LOG)
    print('cnnseg average time cost: {:.2f}ms'.format(np.sum(cnnseg_time) / len(cnnseg_time)))
    print('cpdet average time cost: {:.2f}ms'.format(np.sum(cpdet_time) / len(cpdet_time)))

    x = np.arange(len(cnnseg_time))
    y = np.array(cnnseg_time)
    yhat = savgol_filter(y, 51, 3)
    plt.plot(x, y, linestyle=':', linewidth=0.2, color='orange')
    plt.plot(x, yhat, linewidth=1, color='orange', label='cnnseg')

    x = np.arange(len(cpdet_time))
    y = np.array(cpdet_time)
    yhat = savgol_filter(y, 51, 3)
    plt.plot(x, y, linestyle=':', linewidth=0.2, color='green')
    plt.plot(x, yhat, linewidth=1, color='green', label='cpdet fp16')

    plt.legend()
    plt.xlabel('frame')
    plt.ylabel('time_cost /ms')
    plt.show()



if __name__ == '__main__':
    main()