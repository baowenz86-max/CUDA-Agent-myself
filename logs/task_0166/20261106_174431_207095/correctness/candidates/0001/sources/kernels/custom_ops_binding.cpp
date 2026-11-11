#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream
);

extern "C" void log_softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int B,
    int H,
    int W,
    long long stride_b,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream
);

// Helper to validate CUDA float32 tensor
static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// Elementwise ReLU forward
torch::Tensor relu_forward(torch::Tensor x) {
    check_cuda_float32(x, "x");
    auto output = torch::empty_like(x);
    long long numel = x.numel();
    if (numel == 0) {
        return output;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );
    return output;
}

// Log softmax along dim=1 forward
torch::Tensor log_softmax_dim1_forward(torch::Tensor x) {
    check_cuda_float32(x, "x");
    TORCH_CHECK(x.dim() == 3, "log_softmax_dim1_forward expects a 3D tensor [B, H, W]");
    auto output = torch::empty_like(x);
    
    int B = static_cast<int>(x.size(0));
    int H = static_cast<int>(x.size(1));
    int W = static_cast<int>(x.size(2));
    
    long long stride_b = x.stride(0);
    long long stride_h = x.stride(1);
    long long stride_w = x.stride(2);
    
    // Ensure strides are positive (contiguous)
    TORCH_CHECK(stride_b > 0 && stride_h > 0 && stride_w > 0, "x must have positive strides");
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        H,
        W,
        stride_b,
        stride_h,
        stride_w,
        stream
    );
    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward (CUDA)",
          pybind11::arg("x"));
    m.def("log_softmax_dim1_forward", &log_softmax_dim1_forward, "Log softmax along dim=1 (CUDA)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
