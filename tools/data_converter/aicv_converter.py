# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
from pathlib import Path

import mmcv
import json
import numpy as np
from tqdm import tqdm
from tools.visualizer import pypcd
from mmdet3d.core.bbox import box_np_ops
from .kitti_data_utils import get_aicv_image_info

def _read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line for line in lines]

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


class AICV2KITTI(object):
    """AICV to KITTI converter.

    This class serves as the converter to change the aicv raw data to KITTI
    format.

    Args:
        load_dir (str): Directory to load aicv raw data.
        save_dir (str): Directory to save data in KITTI format.
        prefix (str): Prefix of filename. In general, 0 for training, 1 for
            validation and 2 for testing.
        workers (int, optional): Number of workers for the parallel process.
        test_mode (bool, optional): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 load_dir,
                 save_dir,
                 test_mode=False, 
                 workers=64):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_aicv_locations = None
        self.save_track_id = False

        self.type_list = [
            'smallMot', 'bigMot', 'OnlyTricycle', 'OnlyBicycle', 
            'Tricyclist', 'bicyclist', 'motorcyclist', 'pedestrian', 
            'TrafficCone', 'others', 'fog', 'stopBar', 'smallMovable', 
            'smallUnmovable', 'crashBarrel', 'safetyBarrier', 'sign'
        ]

        self.aicv_to_kitti_class_map = {
            'smallMot': 'Car',
            'bigMot': 'Car',
            'OnlyTricycle': 'Car',
            'OnlyBicycle': 'Car',
            'Tricyclist': 'Cyclist',
            'bicyclist': 'Cyclist',
            'motorcyclist': 'Cyclist',
            'pedestrian': 'Pedestrian',
            'TrafficCone': 'Sign', 
            'stopBar': 'Sign', 
            'crashBarrel': 'Sign', 
            'safetyBarrier': 'Sign', 
            'sign': 'Sign', 
            'smallMovable': 'DontCare',
            'smallUnmovable': 'DontCare', 
            'fog': 'DontCare',
            'others': 'DontCare'
        }
        self.selected_kitti_classes = ['Car', 'Cyclist', 'Pedestrian', 'Sign']
        
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.test_mode = test_mode
        self.workers = int(workers)

        self.result_pathnames = _read_file(load_dir + f'/result.txt')[1:]

        self.label_save_dir = f'{self.save_dir}/label'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        for frame_idx in tqdm(range(len(self))):
            infos = json.loads(self.result_pathnames[frame_idx].split('\t')[1])
            pcd_pathname = osp.join(
                self.load_dir, 
                infos['datasetsRelatedFiles'][0]['localRelativePath'], 
                infos['datasetsRelatedFiles'][0]['fileName'])
            annotations = infos['labelData']['result']
            poses = infos['poses']['velodyne_points'].split(' ')
            pose = [float(val) for val in poses[2:5]]
            timestamp = infos['frameTimestamp']

            self.save_lidar(pcd_pathname, frame_idx)
            self.save_pose(pose, frame_idx)
            self.save_timestamp(timestamp, frame_idx)
            self.save_label(annotations, frame_idx)
        print('\nFinished ...')

    # def convert(self):
    #     """Convert action."""
    #     print('Start converting ...')
    #     mmcv.track_parallel_progress(self.convert_one, range(len(self)),
    #                                  self.workers)
    #     print('\nFinished ...')

    def convert_one(self, frame_idx):
        infos = json.loads(self.result_pathnames[frame_idx].split('\t')[1])
        pcd_pathname = osp.join(
            self.load_dir, 
            infos['datasetsRelatedFiles'][0]['localRelativePath'], 
            infos['datasetsRelatedFiles'][0]['fileName'])
        annotations = infos['labelData']['result']
        poses = infos['poses']['velodyne_points'].split(' ')
        pose = [float(val) for val in poses[2:5]]
        timestamp = infos['frameTimestamp']

        self.save_lidar(pcd_pathname, frame_idx)
        self.save_pose(pose, frame_idx)
        self.save_timestamp(timestamp, frame_idx)
        self.save_label(annotations, frame_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.result_pathnames)

    def save_lidar(self, pcd_pathname, frame_idx):
        point_cloud_path = f'{self.point_cloud_save_dir}/{str(frame_idx).zfill(6)}.bin'

        pcd = pypcd.PointCloud.from_path(pcd_pathname)
        point_cloud = np.stack([pcd.pc_data['x'], pcd.pc_data['y'], 
                                pcd.pc_data['z'], pcd.pc_data['intensity']]).transpose(1, 0)
        point_cloud.astype(np.float32).tofile(point_cloud_path)

    def save_label(self, annotations, frame_idx):
        """Parse and save the label data in txt format.
        The relation between aicv and kitti coordinates is noteworthy:
        1. x, y, z correspond to l, w, h (aicv) -> w, h, l
        3. bbox origin at volumetric center (aicv) -> bottom center (kitti)

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        fp_label = open(
            f'{self.label_save_dir}/{str(frame_idx).zfill(6)}.txt', 'w+')
        fp_label.close()
        for annotation in annotations:
            position = annotation['position']
            rotation = annotation['rotation']
            type = annotation['type']
            size = annotation['size']

            type = self.aicv_to_kitti_class_map[type]
            if type not in self.selected_kitti_classes:
                continue

            # not available
            truncated = 0
            occluded = 0
            alpha = -10
            bounding_box = [0, 0, 100, 100]

            length = size[0]
            width = size[1]
            height = size[2]
            x = position['x']
            y = position['y']
            z = position['z'] - height / 2
            rotation_y = rotation['phi']
            # [w, h, l] will transfose to [l, w, h] in get_label_anno() of kitti_data_utils.py
            line = type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(width, 2), round(height, 2), round(length, 2),
                    round(x, 2), round(y, 2), round(z, 2),
                    round(rotation_y, 2))

            fp_label = open(
                f'{self.label_save_dir}/{str(frame_idx).zfill(6)}.txt', 'a')
            fp_label.write(line)
            fp_label.close()

    def save_timestamp(self, timestamp, frame_idx):
        """Save the timestamp data in a separate file instead of the
        pointcloud.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        with open(osp.join(f'{self.timestamp_save_dir}/{str(frame_idx).zfill(6)}.txt'), 'w') as f:
            f.write(str(timestamp))

    def save_pose(self, pose, frame_idx):
        np.savetxt(
            osp.join(f'{self.pose_save_dir}/{str(frame_idx).zfill(6)}.txt'),
            np.array(pose))

    def create_folder(self):
        """Create folder for data preprocessing."""
        dir_list = [
            self.point_cloud_save_dir, 
            self.pose_save_dir, 
            self.label_save_dir, 
            self.timestamp_save_dir
        ]
        for d in dir_list:
            mmcv.mkdir_or_exist(d)


def _calculate_num_points_in_gt(data_path,
                                      infos,
                                      relative_path,
                                      num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['point_cloud']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_lidar = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)

def create_aicv_info_file(data_path,
                          pkl_prefix='aicv',
                          save_path=None,
                          relative_path=True,
                          max_sweeps=5,
                          workers=8):
    """Create info file of aicv dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str, optional): Prefix of the info file to be generated.
            Default: 'aicv'.
        save_path (str, optional): Path to save the info file.
            Default: None.
        relative_path (bool, optional): Whether to use relative path.
            Default: True.
        max_sweeps (int, optional): Max sweeps before the detection frame
            to be used. Default: 5.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    aicv_infos_train = get_aicv_image_info(
        data_path,
        training=True,
        velodyne=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, aicv_infos_train, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Aicv info train file is saved to {filename}')
    mmcv.dump(aicv_infos_train, filename)
    
    aicv_infos_val = get_aicv_image_info(
        data_path,
        training=True,
        velodyne=True,
        image_ids=val_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, aicv_infos_val, relative_path)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Aicv info val file is saved to {filename}')
    mmcv.dump(aicv_infos_val, filename)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Aicv info trainval file is saved to {filename}')
    mmcv.dump(aicv_infos_train + aicv_infos_val, filename)