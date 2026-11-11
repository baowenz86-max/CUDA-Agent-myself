// PyTorch C++ binding for custom LRN CUDA kernel
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declaration of launcher from custom_ops.cu
extern "C" void lrn_forward_launcher(
    const float* x,
    float* y,
    int N,
    int C,
    int H,
    int W,
    int stride_n,
    int stride_c,
    int stride_h,
    int stride_w,
    int out_stride_n,
    int out_stride_c,
    int out_stride_h,
    int out_stride_w,
    int size,
    float alpha,
    float beta,
    float k,
    cudaStream_t stream);

torch::Tensor lrn_forward(
    torch::Tensor x,
    int64_t size,
    double alpha,
    double beta,
    double k) {
  // Input validation
  TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
  TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

  // Allocate output with same shape and properties
  auto y = torch::empty_like(x);

  // Get dimensions
  int N = static_cast<int>(x.size(0));
  int C = static_cast<int>(x.size(1));
  int H = static_cast<int>(x.size(2));
  int W = static_cast<int>(x.size(3));

  // Get strides (elements)
  int stride_n = static_cast<int>(x.stride(0));
  int stride_c = static_cast<int>(x.stride(1));
  int stride_h = static_cast<int>(x.stride(2));
  int stride_w = static_cast<int>(x.stride(3));

  int out_stride_n = static_cast<int>(y.stride(0));
  int out_stride_c = static_cast<int>(y.stride(1));
  int out_stride_h = static_cast<int>(y.stride(2));
  int out_stride_w = static_cast<int>(y.stride(3));

  // Get current CUDA stream
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  // Call launcher
  lrn_forward_launcher(
      x.data_ptr<float>(),
      y.data_ptr<float>(),
      N, C, H, W,
      stride_n, stride_c, stride_h, stride_w,
      out_stride_n, out_stride_c, out_stride_h, out_stride_w,
      static_cast<int>(size),
      static_cast<float>(alpha),
      static_cast<float>(beta),
      static_cast<float>(k),
      stream);

  // Check for CUDA errors after launch
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

  return y;
}

void register_lrn_forward(pybind11::module& m) {
  m.def("lrn_forward", &lrn_forward,
        "Local Response Normalization forward (CUDA)",
        py::arg("x"),
        py::arg("size"),
        py::arg("alpha"),
        py::arg("beta"),
        py::arg("k"));
}

REGISTER_BINDING(lrn_forward, register_lrn_forward);
