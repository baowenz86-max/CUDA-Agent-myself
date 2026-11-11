// PyTorch binding for custom ReLU and cumsum dim=1 kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launchers from custom_ops.cu
extern "C" void custom_relu_launcher(float* output, const float* input,
                                     long long numel, cudaStream_t stream);
extern "C" void custom_cumsum_dim1_launcher(float* output, const float* input,
                                            int B, int C, int H, int W,
                                            cudaStream_t stream);

// Helper to check CUDA errors after launch
static void check_cuda_error(const char* op_name) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, op_name, " CUDA error: ", cudaGetErrorString(err));
}

// ReLU forward: elementwise max(x, 0)
torch::Tensor custom_relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_relu_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_relu_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "custom_relu_forward: input must be 4D");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    long long numel = x.numel();
    custom_relu_launcher(output.data_ptr<float>(), x.data_ptr<float>(), numel, stream);
    check_cuda_error("custom_relu_launcher");

    return output;
}

// Cumsum along dim=1 for [B, C, H, W]
torch::Tensor custom_cumsum_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_cumsum_dim1_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_cumsum_dim1_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_cumsum_dim1_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "custom_cumsum_dim1_forward: input must be 4D");

    auto output = torch::empty_like(x);

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_cumsum_dim1_launcher(output.data_ptr<float>(), x.data_ptr<float>(),
                                B, C, H, W, stream);
    check_cuda_error("custom_cumsum_dim1_launcher");

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_relu_forward", &custom_relu_forward,
          "ReLU forward (custom CUDA)",
          py::arg("x"));
    m.def("custom_cumsum_dim1_forward", &custom_cumsum_dim1_forward,
          "Cumsum along dim=1 forward (custom CUDA)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
