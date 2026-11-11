#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void relu_cumsum_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t in_strides_n, int64_t in_strides_c, int64_t in_strides_h, int64_t in_strides_w,
    int64_t out_strides_n, int64_t out_strides_c, int64_t out_strides_h, int64_t out_strides_w,
    cudaStream_t stream);

// PyTorch wrapper for fused relu + cumsum(dim=1)
torch::Tensor relu_cumsum_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Expected 4D tensor (N, C, H, W)");

    auto N = x.size(0);
    auto C = x.size(1);
    auto H = x.size(2);
    auto W = x.size(3);

    // Output same shape, same dtype/device
    auto output = torch::empty(x.sizes(), x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch CUDA kernel
    relu_cumsum_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_relu_cumsum(pybind11::module& m) {
    m.def("relu_cumsum_forward", &relu_cumsum_forward,
          "Fused ReLU and cumulative sum along dim=1",
          py::arg("x"));
}

REGISTER_BINDING(relu_cumsum, register_relu_cumsum);
