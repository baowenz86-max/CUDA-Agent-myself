#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void var_mean_dim_launcher(
    const float* x,
    float* var_out,
    float* mean_out,
    int64_t outer_size,
    int64_t inner_size,
    int64_t stride_x_outer,
    int64_t stride_x_inner,
    int64_t stride_out_outer,
    cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor> var_mean_dim(
    torch::Tensor x, int64_t dim, bool keepdim) {
  // 输入检查
  TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA device");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
  TORCH_CHECK(x.dim() > 0, "Input tensor must have at least one dimension");

  // 归一化 dim 为正值
  int64_t ndim = x.dim();
  if (dim < 0) dim += ndim;
  TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

  // 计算输出形状：如果 keepdim 为 true，则保留维度并将尺寸设为 1，否则移除该维度
  std::vector<int64_t> out_shape;
  for (int64_t i = 0; i < ndim; ++i) {
    if (i == dim) {
      if (keepdim) {
        out_shape.push_back(1);
      }
    } else {
      out_shape.push_back(x.size(i));
    }
  }

  auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
  auto var_out = torch::empty(out_shape, options);
  auto mean_out = torch::empty(out_shape, options);

  // 计算 launcher 参数
  int64_t outer_size = 1;
  for (int64_t i = 0; i < dim; ++i) {
    outer_size *= x.size(i);
  }
  int64_t inner_size = x.size(dim);

  // 获取步长信息
  int64_t stride_x_inner = 1;
  int64_t stride_x_outer = x.stride(dim);
  if (dim < ndim - 1) {
    stride_x_inner = x.stride(dim + 1);
  }
  int64_t stride_out_outer = 1;

  // 获取当前 CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  var_mean_dim_launcher(
      x.data_ptr<float>(),
      var_out.data_ptr<float>(),
      mean_out.data_ptr<float>(),
      outer_size,
      inner_size,
      stride_x_outer,
      stride_x_inner,
      stride_out_outer,
      stream);

  // 检查 CUDA 错误
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return std::make_tuple(var_out, mean_out);
}

void register_var_mean_dim(pybind11::module& m) {
  m.def("var_mean_dim", &var_mean_dim,
        "Compute variance and mean along a dimension",
        py::arg("x"),
        py::arg("dim"),
        py::arg("keepdim"));
}

REGISTER_BINDING(var_mean_dim, register_var_mean_dim);
