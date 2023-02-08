#pragma once
#include <torch/extension.h>

namespace voxelization {

void custom_dynamic_voxelize_cpu(const at::Tensor &points,
                                 at::Tensor &coors,
                                 at::Tensor &bev_features,
                                 at::Tensor &coor_to_voxelidx,
                                 const std::vector<float> voxel_size,
                                 const std::vector<float> coors_range,
                                 const int NDim = 3);

#ifdef WITH_CUDA
void custom_dynamic_voxelize_gpu(const at::Tensor &points, 
                                 at::Tensor &coors,
                                 at::Tensor &bev_features,
                                 at::Tensor &coor_to_voxelidx,
                                 const std::vector<float> voxel_size,
                                 const std::vector<float> coors_range,
                                 const int NDim = 3);
#endif

// Interface for Python
inline void custom_dynamic_voxelize(const at::Tensor &points, 
                                    at::Tensor &coors,
                                    at::Tensor &bev_features,
                                    at::Tensor &coor_to_voxelidx,
                                    const std::vector<float> voxel_size,
                                    const std::vector<float> coors_range,
                                    const int NDim = 3) {
  if (points.device().is_cuda()) {
#ifdef WITH_CUDA
    return custom_dynamic_voxelize_gpu(points,
                                       coors,
                                       bev_features,
                                       coor_to_voxelidx,
                                       voxel_size,
                                       coors_range,
                                       NDim);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return custom_dynamic_voxelize_cpu(points,
                                     coors,
                                     bev_features,
                                     coor_to_voxelidx,
                                     voxel_size,
                                     coors_range,
                                     NDim);
}

}  // namespace voxelization
