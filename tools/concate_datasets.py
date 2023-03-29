import os
import os.path as osp
import shutil
import mmcv
import argparse

DATASETS = ['adt-2019', 'concat-m-r-t']
DST = 'concat-hesai40'
DATA_ROOT = 'data/hesai40'


def read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line for line in lines]

def read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

def get_dataset_capacity(dataset):
    trainval = read_file(
        osp.join(DATA_ROOT, dataset, 'kitti_format/ImageSets/trainval.txt'))
    return len(trainval)

def write_concat_imagesets(count, dataset, filename):
    dst_path = osp.join(DATA_ROOT, DST, 'kitti_format/ImageSets', f'{filename}.txt')
    idx = read_imageset_file(
        osp.join(DATA_ROOT, dataset, 'kitti_format/ImageSets', f'{filename}.txt'))
    with open(dst_path, 'a+') as f:
        for i in idx:
            f.write('{:06d}\n'.format(count + i))

def copy_or_link_file(count, length, dataset, dirname, mode='link'):
    # 没有文件夹则创建
    mmcv.mkdir_or_exist(osp.join(DATA_ROOT, DST, f'kitti_format/training/{dirname}'))

    suffix = '.bin' if dirname == 'velodyne' else '.txt'
    for i in range(length):
        src_file = osp.join(DATA_ROOT, dataset, 'kitti_format/training/{}/{:06d}{}'.format(dirname, i, suffix))
        dst_file = osp.join(DATA_ROOT, DST, 'kitti_format/training/{}/{:06d}{}'.format(dirname, count + i, suffix))
        if mode == 'link':
            src_file = osp.join('../../../..', dataset, 'kitti_format/training/{}/{:06d}{}'.format(dirname, i, suffix))
            os.symlink(src_file, dst_file)
        elif mode == 'copy':
            shutil.copyfile(src_file, dst_file)
        else:
            print('Error: only support link or copy')
            exit()


parser = argparse.ArgumentParser(description='Data converter arg parser')
# parser.add_argument('data_root', metavar='data/hesai40', help='Upper level path of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument('--workers', type=int, default=8, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    # 删除 DST 下的旧 ImageSets 并创建新的
    dst_imagesets_dir = osp.join(DATA_ROOT, DST, 'kitti_format/ImageSets')
    if osp.exists(dst_imagesets_dir):
        shutil.rmtree(dst_imagesets_dir)
    mmcv.mkdir_or_exist(dst_imagesets_dir)

    count = 0
    for dataset in DATASETS:
        # 获取数据集数据量
        length = get_dataset_capacity(dataset)
        print('dataset {} capacity: {}'.format(dataset, length))

        # 生成 ImageSets
        write_concat_imagesets(count, dataset, 'train')
        write_concat_imagesets(count, dataset, 'val')

        # 复制或者创建文件链接 label/pose/timestamp/velodyne
        copy_or_link_file(count, length, dataset, 'label', mode='link')
        copy_or_link_file(count, length, dataset, 'pose', mode='link')
        copy_or_link_file(count, length, dataset, 'timestamp', mode='link')
        copy_or_link_file(count, length, dataset, 'velodyne', mode='link')
        count += length

    # 生成 trainval.txt
    trainval_idx = [*range(count)]
    with open(osp.join(DATA_ROOT, DST, 'kitti_format/ImageSets/trainval.txt'), 'w+') as f:
        for idx in trainval_idx:
            # f.write(str(idx) + '\n')
            f.write('{:06d}\n'.format(idx))

    # 生成 pkl
    from tools.data_converter import aicv_converter as aicv
    out_dir = osp.join(DATA_ROOT, DST, 'kitti_format')
    extra_tag = DST
    aicv.create_aicv_info_file(
        out_dir, pkl_prefix=extra_tag, workers=args.workers)

    # 生成 gt_database
