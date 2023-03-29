# Copyright (c) OpenMMLab. All rights reserved.
import numba
import numpy as np
import time


class BevFeatureGenerator(object):
    """Bev feature generator in numpy implementation.

    Args:
        voxel_size (list[float]): Size of a single voxel
        point_cloud_range (list[float]): Range of points
    """

    def __init__(self, voxel_size, point_cloud_range):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._grid_size = grid_size

    def generate(self, points):
        """Generate bev_map_feature given points."""
        return points_to_voxel(points,
                               self._voxel_size,
                               self._point_cloud_range,
                               True)

    @property
    def voxel_size(self):
        """list[float]: Size of a single voxel."""
        return self._voxel_size

    @property
    def point_cloud_range(self):
        """list[float]: Range of point cloud."""
        return self._point_cloud_range

    @property
    def grid_size(self):
        """np.ndarray: The size of grids."""
        return self._grid_size

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        indent = ' ' * (len(repr_str) + 1)
        repr_str += f'(voxel_size={self._voxel_size},\n'
        repr_str += indent + 'point_cloud_range='
        repr_str += f'{self._point_cloud_range.tolist()},\n'
        repr_str += indent + f'grid_size={self._grid_size.tolist()}'
        repr_str += ')'
        return repr_str


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    reverse_index=True):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Voxel range.
            format: xyzxyz, minmax
        max_points (int): Indicate maximum points contained in a voxel.
        reverse_index (bool): Whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels (int): Maximum number of voxels this function creates.
            For second, 20000 is a good choice. Points should be shuffled for
            randomness before this function because max_voxels drops points.

    Returns:
        tuple[np.ndarray]:
            voxels: [M, max_points, ndim] float tensor. only contain points.
            coordinates: [M, 3] int32 tensor.
            num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    if reverse_index:
        voxelmap_shape = voxelmap_shape[::-1]
    # don't create large array in jit(nopython=True) code.
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)

    # (top_z, mean_z, pts_count, top_intensity, mean_intensity, nonempty height_bin)
    if reverse_index:
        bev_map_feature_shape = voxelmap_shape[1:]  # zyx
    else:
        bev_map_feature_shape = voxelmap_shape[:-1] # xyz
    # [16, 848, 848]
    height_bin_num = int((2 - (-3)) / 0.5)          # 0.5m 10
    bev_map_feature = np.zeros(
        shape=((6 + height_bin_num, ) + bev_map_feature_shape), dtype=points.dtype)

    if reverse_index:
        tic = time.perf_counter()
        voxel_num = _points_to_voxel_reverse_kernel(
            points, voxel_size, coors_range, coor_to_voxelidx, bev_map_feature)
        toc = time.perf_counter()
        print('time cost {:.2f}ms'.format((toc-tic)*1000))
    else:
        voxel_num = _points_to_voxel_kernel(points, voxel_size, coors_range,
                                            coor_to_voxelidx, bev_map_feature)
    
    invalid_idx = bev_map_feature[2, ...] < 1.0
    valid_idx = bev_map_feature[2, ...] >= 1.0
    bev_map_feature[2, invalid_idx] = 1
    bev_map_feature[1, ...] = bev_map_feature[1, ...] / (bev_map_feature[2, ...])
    bev_map_feature[4, ...] = bev_map_feature[4, ...] / (bev_map_feature[2, ...])
    bev_map_feature[2, valid_idx] += 1
    bev_map_feature[2, ...] = np.log(bev_map_feature[2, ...])

    return bev_map_feature


@numba.jit(nopython=True)
def _points_to_voxel_reverse_kernel(points,
                                    voxel_size,
                                    coors_range,
                                    coor_to_voxelidx,
                                    bev_map_feature):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        coor_to_voxelidx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # reduce performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    ndim_minus_1 = ndim - 1
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # np.round(grid_size)
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False

    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1: # first point
            bev_map_feature[0, coor[1], coor[2]] = points[i, 2]
            bev_map_feature[3, coor[1], coor[2]] = points[i, 3] / 255.0
            bev_map_feature[1, coor[1], coor[2]] = bev_map_feature[1, coor[1], coor[2]] + points[i, 2]
            bev_map_feature[2, coor[1], coor[2]] += 1
            bev_map_feature[4, coor[1], coor[2]] = bev_map_feature[4, coor[1], coor[2]] + points[i, 3] / 255.0
            bev_map_feature[5, coor[1], coor[2]] = 1.0

            voxelidx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
        else:
            if points[i][2] > bev_map_feature[0, coor[1], coor[2]]:
                bev_map_feature[0, coor[1], coor[2]] = points[i, 2]
                bev_map_feature[3, coor[1], coor[2]] = points[i, 3] / 255.0
            bev_map_feature[1, coor[1], coor[2]] = bev_map_feature[1, coor[1], coor[2]] + points[i, 2]
            bev_map_feature[2, coor[1], coor[2]] += 1
            bev_map_feature[4, coor[1], coor[2]] = bev_map_feature[4, coor[1], coor[2]] + points[i, 3] / 255.0

        height_bin_index = int(np.floor((points[i, 2] - (-3)) / 0.5))
        height_bin_index = 0 if height_bin_index < 0 else height_bin_index
        height_bin_index = 9 if height_bin_index > 9 else height_bin_index
        bev_map_feature[6 + height_bin_index, coor[1], coor[2]] = 1.0
    return voxel_num


@numba.jit(nopython=True)
def _points_to_voxel_kernel(points,
                            voxel_size,
                            coors_range,
                            coor_to_voxelidx,
                            bev_map_feature):
    """convert kitti points(N, >=3) to voxels.

    Args:
        points (np.ndarray): [N, ndim]. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size (list, tuple, np.ndarray): [3] xyz, indicate voxel size.
        coors_range (list[float | tuple[float] | ndarray]): Range of voxels.
            format: xyzxyz, minmax
        coor_to_voxel_idx (np.ndarray): A voxel grid of shape (D, H, W),
            which has the same shape as the complete voxel map. It indicates
            the index of each corresponding voxel.

    Returns:
        tuple[np.ndarray]:
            voxels: Shape [M, max_points, ndim], only contain points.
            coordinates: Shape [M, 3].
            num_points_per_voxel: Shape [M].
    """
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    # lower_bound = coors_range[:3]
    # upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    failed = False

    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1:
            bev_map_feature[0, coor[0], coor[1]] = points[i, 2]
            bev_map_feature[3, coor[0], coor[1]] = points[i, 3] / 255.0
            bev_map_feature[1, coor[0], coor[1]] = bev_map_feature[1, coor[0], coor[1]] + points[i, 2]
            bev_map_feature[2, coor[0], coor[1]] += 1
            bev_map_feature[4, coor[0], coor[1]] = bev_map_feature[4, coor[0], coor[1]] + points[i, 3] / 255.0
            bev_map_feature[5, coor[0], coor[1]] = 1.0

            height_bin_index = int(np.floor((points[i, 2] - (-3)) / 0.5))
            height_bin_index = 0 if height_bin_index < 0 else height_bin_index
            height_bin_index = 9 if height_bin_index > 9 else height_bin_index
            bev_map_feature[6 + height_bin_index, coor[0], coor[1]] = 1.0
            voxelidx = voxel_num
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
        else:
            if points[i, 2] > bev_map_feature[0, coor[0], coor[1]]:
                bev_map_feature[0, coor[0], coor[1]] = points[i, 2]
                bev_map_feature[3, coor[0], coor[1]] = points[i, 3] / 255.0
            bev_map_feature[1, coor[0], coor[1]] = bev_map_feature[1, coor[0], coor[1]] + points[i, 2]
            bev_map_feature[2, coor[0], coor[1]] += 1
            bev_map_feature[4, coor[0], coor[1]] = bev_map_feature[4, coor[0], coor[1]] + points[i, 3] / 255.0

            height_bin_index = int(np.floor((points[i, 2] - (-3)) / 0.5))
            height_bin_index = 0 if height_bin_index < 0 else height_bin_index
            height_bin_index = 9 if height_bin_index > 9 else height_bin_index
            bev_map_feature[6 + height_bin_index, coor[0], coor[1]] = 1.0
    return voxel_num
