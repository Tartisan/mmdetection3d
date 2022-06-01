# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import mmcv
import numpy as np
import json
from tools.visualizer import pypcd

def _read_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line for line in lines]


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
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        self.selected_aicv_classes = ['smallMot', 'bigMot', 'OnlyTricycle', 
            'OnlyBicycle', 'Tricyclist', 'bicyclist', 'motorcyclist', 
            'pedestrian']

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
        
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.test_mode = test_mode

        self.result_pathnames = _read_file(load_dir + f'/result.txt')[1:]

        self.label_save_dir = f'{self.save_dir}/label'
        self.point_cloud_save_dir = f'{self.save_dir}/velodyne'
        self.pose_save_dir = f'{self.save_dir}/pose'
        self.timestamp_save_dir = f'{self.save_dir}/timestamp'

        self.create_folder()

    def convert(self):
        """Convert action."""
        print('Start converting ...')
        frame_idx = 0
        for result in self.result_pathnames:
            infos = json.loads(result.split('\t')[1])
            pcd_pathname = osp.join(
                self.load_dir, 
                infos['datasetsRelatedFiles'][0]['localRelativePath'], 
                infos['datasetsRelatedFiles'][0]['fileName'])
            annotations = infos['labelData']['result']
            poses = infos['poses']['velodyne_points'].split(' ')
            pose = [float(val) for val in poses[2:5]]
            timestamp = infos['frameTimestamp']

            self.save_lidar(pcd_pathname, frame_idx)
            self.save_label(annotations, frame_idx)
            self.save_pose(pose, frame_idx)
            self.save_timestamp(timestamp, frame_idx)
            frame_idx = frame_idx + 1

        print('\nFinished convertion {}/{}'.format(frame_idx, len(self)))

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

            if type not in self.selected_aicv_classes:
                continue

            type = self.aicv_to_kitti_class_map[type]

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

            rotation_y = -rotation['phi'] - np.pi / 2
            if rotation_y < -np.pi:
                rotation_y = rotation_y + 2 * np.pi

            # [h, w, l] will transfose to [l, h, w] in get_label_anno() of kitti_data_utils.py:143
            line = type + \
                ' {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n'.format(
                    round(truncated, 2), occluded, round(alpha, 2),
                    round(bounding_box[0], 2), round(bounding_box[1], 2),
                    round(bounding_box[2], 2), round(bounding_box[3], 2),
                    round(height, 2), round(width, 2), round(length, 2),
                    round(-y, 2), round(-z, 2), round(x, 2),
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
