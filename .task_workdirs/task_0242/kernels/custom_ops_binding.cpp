#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_transform_launcher(const float* input,
                                         float* output,
                                         long long numel,
                                         cudaStream_t stream);

extern "C" void smooth_l1_loss_mean_zero_launcher(const float* input,
                                                   float* output,
                                                   long long numel,
                                                   cudaStream_t stream);

// Helper to validate common requirements
static void validate_input(const torch::Tensor& x, const char* op_name) {
    TORCH_CHECK(x.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, op_name, ": input must be float32");
    TORCH_CHECK(x.is_contiguous(), op_name, ": input must be contiguous");
    TORCH_CHECK(x.dim() == 3 && x.size(0) == 512 && x.size(1) == 512 && x.size(2) == 512,
                op_name, ": expected shape [512, 512, 512]");
}

// relu_transform: elementwise ReLU
torch::Tensor relu_transform(torch::Tensor x) {
    validate_input(x, "relu_transform");
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_transform_launcher(x.data_ptr<float>(),
                            output.data_ptr<float>(),
                            static_cast<long long>(x.numel()),
                            stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu_transform kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// smooth_l1_loss_mean_zero: mean smooth L1 loss vs zero targets
torch::Tensor smooth_l1_loss_mean_zero(torch::Tensor x) {
    validate_input(x, "smooth_l1_loss_mean_zero");
    auto output = torch::empty({}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    smooth_l1_loss_mean_zero_launcher(x.data_ptr<float>(),
                                      output.data_ptr<float>(),
                                      static_cast<long long>(x.numel()),
                                      stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "smooth_l1_loss_mean_zero kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu_transform", &relu_transform,
          "Elementwise ReLU (max(x, 0))",
          pybind11::arg("x"));
    m.def("smooth_l1_loss_mean_zero", &smooth_l1_loss_mean_zero,
          "Mean smooth L1 loss between x and zero",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
