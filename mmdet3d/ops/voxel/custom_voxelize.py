# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair

from .voxel_layer import custom_dynamic_voxelize
from mmdet3d.models.builder import VOXEL_LAYERS


class _CustomVoxelization(Function):
    @staticmethod
    def forward(ctx,
                points,
                voxel_size,
                coors_range,
                max_points=35,
                max_voxels=20000,
                grid_size=torch.tensor([848, 848, 1]),
                deterministic=True):
        """convert kitti points(N, >=3) to voxels.

        Args:
            points: [N, ndim] float tensor. points[:, :3] contain xyz points
                and points[:, 3:] contain other information like reflectivity
            voxel_size: [3] list/tuple or array, float. xyz, indicate voxel
                size
            coors_range: [6] list/tuple or array, float. indicate voxel
                range. format: xyzxyz, minmax
            max_points: int. indicate maximum points contained in a voxel. if
                max_points=-1, it means using dynamic_voxelize
            max_voxels: int. indicate maximum voxels this function create.
                for second, 20000 is a good choice. Users should shuffle points
                before call this function because max_voxels may drop points.
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.

        Returns:
            voxels: [M, max_points, ndim] float tensor. only contain points
                    and returned when max_points != -1.
            coordinates: [M, 3] int32 tensor, always returned.
            num_points_per_voxel: [M] int32 tensor. Only returned when
                max_points != -1.
        """
        if max_points == -1 or max_voxels == -1:
            coors = points.new_zeros(size=(points.size(0), 3), dtype=torch.int)

            voxelmap_shape = grid_size[:2]
            # [w, h, d] -> [d, h, w]
            voxelmap_shape = (*voxelmap_shape, 1)[::-1]
            # [16, 848, 848]
            height_bin_num = int((2 - (-3)) / 0.5)          # 0.5m 10
            bev_features = torch.zeros(
                size=((6 + height_bin_num, ) + voxelmap_shape[1:]), 
                device=points.device,
                dtype=points.dtype)
            # [1, 848, 848]
            coor2voxelidx = -torch.ones(size=voxelmap_shape, 
                                        device=points.device, 
                                        dtype=torch.int)
            custom_dynamic_voxelize(points, 
                                    coors,
                                    bev_features,
                                    coor2voxelidx,
                                    voxel_size,
                                    coors_range,
                                    3)
            
            invalid_idx = bev_features[4, ...] < 1.0
            valid_idx = bev_features[4, ...] >= 1.0
            bev_features[4, invalid_idx] = 1
            bev_features[1, ...] = bev_features[1, ...] / (bev_features[4, ...])
            bev_features[3, ...] = bev_features[3, ...] / (bev_features[4, ...])
            bev_features[4, valid_idx] += 1
            bev_features[4, ...] = torch.log(bev_features[4, ...])
            return coors, bev_features
        else:
            print('Error: CustomVoxelization only support dynamic voxelize now')
            return


custom_voxelization = _CustomVoxelization.apply


@VOXEL_LAYERS.register_module()
class CustomVoxelization(nn.Module):

    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=20000,
                 deterministic=True):
        super(CustomVoxelization, self).__init__()
        """
        Args:
            voxel_size (list): list [x, y, z] size of three dimension
            point_cloud_range (list):
                [x_min, y_min, z_min, x_max, y_max, z_max]
            max_num_points (int): max number of points per voxel
            max_voxels (tuple or int): max number of voxels in
                (training, testing) time
            deterministic: bool. whether to invoke the non-deterministic
                version of hard-voxelization implementations. non-deterministic
                version is considerablly fast but is not deterministic. only
                affects hard voxelization. default True. for more information
                of this argument and the implementation insights, please refer
                to the following links:
                https://github.com/open-mmlab/mmdetection3d/issues/894
                https://github.com/open-mmlab/mmdetection3d/pull/904
                it is an experimental feature and we will appreciate it if
                you could share with us the failing cases.
        """
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.max_num_points = max_num_points
        if isinstance(max_voxels, tuple):
            self.max_voxels = max_voxels
        else:
            self.max_voxels = _pair(max_voxels)
        self.deterministic = deterministic

        point_cloud_range = torch.tensor(
            point_cloud_range, dtype=torch.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = torch.tensor(voxel_size, dtype=torch.float32)
        grid_size = (point_cloud_range[3:] -
                     point_cloud_range[:3]) / voxel_size
        grid_size = torch.round(grid_size).long()
        input_feat_shape = grid_size[:2]
        self.grid_size = grid_size
        # the origin shape is as [x-len, y-len, z-len]
        # [w, h, d] -> [d, h, w]
        self.pcd_shape = [*input_feat_shape, 1][::-1]

    def forward(self, input):
        """
        Args:
            input: NC points
        """
        max_voxels = self.max_voxels[0] if self.training else self.max_voxels[1]

        return custom_voxelization(input, 
                                   self.voxel_size,
                                   self.point_cloud_range,
                                   self.max_num_points,
                                   max_voxels,
                                   self.grid_size,
                                   self.deterministic)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'voxel_size=' + str(self.voxel_size)
        tmpstr += ', point_cloud_range=' + str(self.point_cloud_range)
        tmpstr += ', max_num_points=' + str(self.max_num_points)
        tmpstr += ', max_voxels=' + str(self.max_voxels)
        tmpstr += ', deterministic=' + str(self.deterministic)
        tmpstr += ')'
        return tmpstr
