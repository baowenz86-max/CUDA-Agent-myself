// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_equal_launcher(const float* x, const float* y, int64_t numel, cudaStream_t stream, int* host_result);
extern "C" void custom_smooth_l1_loss_launcher(const float* x, const float* y, int64_t numel, float* output, cudaStream_t stream);

// Helper to check CUDA errors
#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                  \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
  } while (0)

// custom_equal: returns bool indicating exact equality of two CUDA float tensors
bool custom_equal(torch::Tensor x, torch::Tensor y) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

  int64_t numel = x.numel();
  if (numel == 0) {
    return true; // empty tensors are equal
  }

  int host_result = 0;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  custom_equal_launcher(x.data_ptr<float>(), y.data_ptr<float>(), numel, stream, &host_result);
  CUDA_CHECK(cudaStreamSynchronize(stream)); // ensure host_result is ready

  return host_result == 0; // launcher writes 0 if equal, 1 otherwise (per .cu logic)
}

// custom_smooth_l1_loss: returns scalar tensor with MEAN smooth L1 loss
torch::Tensor custom_smooth_l1_loss(torch::Tensor x, torch::Tensor y) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
  TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
  TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
  TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

  int64_t numel = x.numel();
  TORCH_CHECK(numel > 0, "input must not be empty");

  auto output = torch::empty({}, x.options()); // scalar tensor on same device and dtype
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  custom_smooth_l1_loss_launcher(x.data_ptr<float>(), y.data_ptr<float>(), numel, output.data_ptr<float>(), stream);
  CUDA_CHECK(cudaGetLastError()); // check launch error

  return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
  m.def("custom_equal", &custom_equal, "Exact element-wise equality of two float32 CUDA tensors",
        py::arg("x"), py::arg("y"));
  m.def("custom_smooth_l1_loss", &custom_smooth_l1_loss, "Smooth L1 loss (mean, beta=1.0) between two float32 CUDA tensors",
        py::arg("x"), py::arg("y"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
