# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import time

from mmdet3d.core.voxel.bev_feature_generator import BevFeatureGenerator


def test_bev_feature_generator():
    np.random.seed(0)
    voxel_size = [0.2, 0.2, 7]
    point_cloud_range = [-84.8, -84.8, -3.5, 84.8, 84.8, 3.5]
    self = BevFeatureGenerator(voxel_size, point_cloud_range)
    points = np.random.rand(150000, 4)
    tic = time.perf_counter()
    bev_map_feature = self.generate(points)
    toc = time.perf_counter()
    print('time cost {:.2f}ms'.format((toc-tic)*1000))
    # voxels, coors, num_points_per_voxel = voxels
    # expected_coors = np.array([[7, 81, 1], [6, 81, 0], [7, 80, 1], [6, 81, 1],
    #                            [7, 81, 0], [6, 80, 1], [7, 80, 0], [6, 80, 0]])
    # expected_num_points_per_voxel = np.array(
    #     [120, 121, 127, 134, 115, 127, 125, 131])
    # assert voxels.shape == (8, 1000, 4)
    # assert np.all(coors == expected_coors)
    # assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

if __name__ == '__main__':
    test_bev_feature_generator()