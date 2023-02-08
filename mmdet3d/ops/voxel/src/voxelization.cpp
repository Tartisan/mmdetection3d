#include <torch/extension.h>
#include "voxelization.h"

namespace voxelization {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("custom_dynamic_voxelize", &custom_dynamic_voxelize, "dynamic voxelization with manual bev features");
}

} // namespace voxelization
