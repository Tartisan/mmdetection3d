# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import time

from mmdet3d.core.voxel.voxel_generator import VoxelGenerator
from mmcv.ops import Voxelization
import torch

def test_voxel_generator():
    np.random.seed(0)
    voxel_size = [0.5, 0.5, 0.5]
    point_cloud_range = [0, -40, -3, 70.4, 40, 1]
    max_num_points = 1000
    self = VoxelGenerator(voxel_size, point_cloud_range, max_num_points)
    points = np.random.rand(1000, 4)
    tic = time.perf_counter()
    voxels = self.generate(points)
    toc = time.perf_counter()
    print('time cost {:.2f}ms'.format((toc-tic)*1000))
    voxels, coors, num_points_per_voxel = voxels
    expected_coors = np.array([[7, 81, 1], [6, 81, 0], [7, 80, 1], [6, 81, 1],
                               [7, 81, 0], [6, 80, 1], [7, 80, 0], [6, 80, 0]])
    expected_num_points_per_voxel = np.array(
        [120, 121, 127, 134, 115, 127, 125, 131])
    assert voxels.shape == (8, 1000, 4)
    assert np.all(coors == expected_coors)
    assert np.all(num_points_per_voxel == expected_num_points_per_voxel)

    pts_voxel_layer = Voxelization(max_num_points=max_num_points, 
                                   voxel_size=voxel_size, 
                                   max_voxels=(20000, 20000),
                                   point_cloud_range=point_cloud_range)
    tic = time.perf_counter()
    voxels = pts_voxel_layer(torch.from_numpy(points))
    toc = time.perf_counter()
    print('time cost {:.2f}ms'.format((toc-tic)*1000))


if __name__ == '__main__':
    test_voxel_generator()