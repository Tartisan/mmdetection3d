#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

#include <ATen/cuda/CUDAApplyUtils.cuh>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

namespace {
int const threadsPerBlock = sizeof(unsigned long long) * 8;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

template <typename T, typename T_int>
__global__ void custom_dynamic_voxelize_kernel(
    const T* points,
    T_int* coors,
    T* bev_features,
    T_int* coor_to_voxelidx,
    const float voxel_x,
    const float voxel_y,
    const float voxel_z,
    const float coors_x_min,
    const float coors_y_min,
    const float coors_z_min,
    const float coors_x_max,
    const float coors_y_max,
    const float coors_z_max,
    const int grid_x,
    const int grid_y,
    const int grid_z,
    const int num_points,
    const int num_features,
    const int NDim) {
  //   const int index = blockIdx.x * threadsPerBlock + threadIdx.x;
  CUDA_1D_KERNEL_LOOP(index, num_points) {
    // To save some computation
    auto points_offset = points + index * num_features;
    auto coors_offset = coors + index * NDim;
    int c_x = floor((points_offset[0] - coors_x_min) / voxel_x);
    if (c_x < 0 || c_x >= grid_x) {
      coors_offset[0] = -1;
      return;
    }

    int c_y = floor((points_offset[1] - coors_y_min) / voxel_y);
    if (c_y < 0 || c_y >= grid_y) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      return;
    }

    int c_z = floor((points_offset[2] - coors_z_min) / voxel_z);
    if (c_z < 0 || c_z >= grid_z) {
      coors_offset[0] = -1;
      coors_offset[1] = -1;
      coors_offset[2] = -1;
      return;
    }

    coors_offset[0] = c_z;
    coors_offset[1] = c_y;
    coors_offset[2] = c_x;
    
    int map_size = grid_x * grid_y;
    int map_index = c_y * grid_x + c_x;
    auto voxelidx = coor_to_voxelidx + c_z * map_size + map_index;
    if (voxelidx[0] == -1) { // first point
      bev_features[0 * map_size + map_index] = points_offset[2];
      bev_features[2 * map_size + map_index] = points_offset[3] / 255.0;
      voxelidx[0] = 0;
    } else {
      if (points_offset[2] > bev_features[0 * map_size + map_index]) {
        bev_features[0 * map_size + map_index] = points_offset[2];
        bev_features[2 * map_size + map_index] = points_offset[3] / 255.0;
      }
    }
    atomicAdd(&bev_features[1 * map_size + map_index], points_offset[2]);
    atomicAdd(&bev_features[3 * map_size + map_index], points_offset[3] / 255.0);
    atomicAdd(&bev_features[4 * map_size + map_index], 1);
    bev_features[5 * map_size + map_index] = 1;

    int height_bin_index = floor((points_offset[2] - (-3)) / 0.5);
    if (height_bin_index < 0) {
      height_bin_index = 0;
    } else if (height_bin_index > 9) {
      height_bin_index = 9;
    }
    bev_features[(6 + height_bin_index) * map_size + map_index] = 1;
  }
}

namespace voxelization {

void custom_dynamic_voxelize_gpu(const at::Tensor& points, 
                                 at::Tensor& coors,
                                 at::Tensor& bev_features,
                                 at::Tensor &coor_to_voxelidx,
                                 const std::vector<float> voxel_size,
                                 const std::vector<float> coors_range,
                                 const int NDim = 3) {
  // current version tooks about 0.04s for one frame on cpu
  // check device
  CHECK_INPUT(points);

  at::cuda::CUDAGuard device_guard(points.device());

  const int num_points = points.size(0);
  const int num_features = points.size(1);

  const float voxel_x = voxel_size[0];
  const float voxel_y = voxel_size[1];
  const float voxel_z = voxel_size[2];
  const float coors_x_min = coors_range[0];
  const float coors_y_min = coors_range[1];
  const float coors_z_min = coors_range[2];
  const float coors_x_max = coors_range[3];
  const float coors_y_max = coors_range[4];
  const float coors_z_max = coors_range[5];

  const int grid_x = round((coors_x_max - coors_x_min) / voxel_x);
  const int grid_y = round((coors_y_max - coors_y_min) / voxel_y);
  const int grid_z = round((coors_z_max - coors_z_min) / voxel_z);

  // auto voxel_num = at::zeros({1}, points.options().dtype(at::kInt));

  const int col_blocks = at::cuda::ATenCeilDiv(num_points, threadsPerBlock);
  dim3 blocks(col_blocks);
  dim3 threads(threadsPerBlock);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_ALL_TYPES(points.scalar_type(), "custom_dynamic_voxelize_kernel", [&] {
    custom_dynamic_voxelize_kernel<scalar_t, int><<<blocks, threads, 0, stream>>>(
        points.contiguous().data_ptr<scalar_t>(),
        coors.contiguous().data_ptr<int>(),
        bev_features.contiguous().data_ptr<scalar_t>(),
        coor_to_voxelidx.contiguous().data_ptr<int>(),
        voxel_x,
        voxel_y,
        voxel_z,
        coors_x_min,
        coors_y_min,
        coors_z_min,
        coors_x_max,
        coors_y_max,
        coors_z_max,
        grid_x,
        grid_y,
        grid_z,
        num_points,
        num_features,
        NDim);
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());

  return;
}

}  // namespace voxelization
