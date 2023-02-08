#include <ATen/TensorUtils.h>
#include <torch/extension.h>
// #include "voxelization.h"

namespace {

template <typename T, typename T_int>
void custom_dynamic_voxelize_kernel(const torch::TensorAccessor<T, 2> points,
                                    torch::TensorAccessor<T_int, 2> coors,
                                    torch::TensorAccessor<T, 3> bev_features,
                                    torch::TensorAccessor<T_int, 3> coor_to_voxelidx,
                                    const std::vector<float> voxel_size,
                                    const std::vector<float> coors_range,
                                    const std::vector<int> grid_size,
                                    const int num_points,
                                    const int num_features,
                                    const int NDim) {
  const int ndim_minus_1 = NDim - 1;
  bool failed = false;
  int voxel_num = 0; // record voxel
  // int coor[NDim];
  int* coor = new int[NDim]();
  int c;

  for (int i = 0; i < num_points; ++i) {
    failed = false;
    for (int j = 0; j < NDim; ++j) {
      c = floor((points[i][j] - coors_range[j]) / voxel_size[j]);
      // necessary to rm points out of range
      if ((c < 0 || c >= grid_size[j])) {
        failed = true;
        break;
      }
      coor[ndim_minus_1 - j] = c;
    }

    for (int k = 0; k < NDim; ++k) {
      if (failed)
        coors[i][k] = -1;
      else
        coors[i][k] = coor[k];
    }

    if (failed) {
      continue;
    }
    int voxelidx = coor_to_voxelidx[coor[0]][coor[1]][coor[2]];
    if (voxelidx == -1) {
      bev_features[0][coor[1]][coor[2]] = points[i][2];
      bev_features[2][coor[1]][coor[2]] = points[i][3] / 255.0;
      coor_to_voxelidx[coor[0]][coor[1]][coor[2]] = voxel_num;
      ++voxel_num;
    } else {
      if (points[i][2] > bev_features[0][coor[1]][coor[2]]) {
        bev_features[0][coor[1]][coor[2]] = points[i][2];
        bev_features[2][coor[1]][coor[2]] = points[i][3] / 255.0;
      }
    }
    bev_features[1][coor[1]][coor[2]] += points[i][2];
    bev_features[3][coor[1]][coor[2]] += points[i][3] / 255.0;
    bev_features[4][coor[1]][coor[2]] += 1;
    bev_features[5][coor[1]][coor[2]] = 1;

    int height_bin_index = floor((points[i][2] - (-3)) / 0.5);
    if (height_bin_index < 0) {
      height_bin_index = 0;
    } else if (height_bin_index > 9) {
      height_bin_index = 9;
    }
    bev_features[6 + height_bin_index][coor[1]][coor[2]] = 1;
  }

  delete[] coor;
  return;
}

}  // namespace

namespace voxelization {

void custom_dynamic_voxelize_cpu(const at::Tensor& points, 
                                 at::Tensor& coors,
                                 at::Tensor &bev_features,
                                 at::Tensor &coor_to_voxelidx,
                                 const std::vector<float> voxel_size,
                                 const std::vector<float> coors_range,
                                 const int NDim = 3) {
  // check device
  AT_ASSERTM(points.device().is_cpu(), "points must be a CPU tensor");

  std::vector<int> grid_size(NDim);
  const int num_points = points.size(0);
  const int num_features = points.size(1);

  for (int i = 0; i < NDim; ++i) {
    grid_size[i] =
        round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
  }

  // coors, num_points_per_voxel, coor_to_voxelidx are int Tensor
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      points.scalar_type(), "hard_voxelize_forward", [&] {
        custom_dynamic_voxelize_kernel<scalar_t, int>(
            points.accessor<scalar_t, 2>(), 
            coors.accessor<int, 2>(),
            bev_features.accessor<scalar_t, 3>(),
            coor_to_voxelidx.accessor<int, 3>(),
            voxel_size,
            coors_range,
            grid_size,
            num_points,
            num_features,
            NDim);
      });

  return;
}

}  // namespace voxelization
