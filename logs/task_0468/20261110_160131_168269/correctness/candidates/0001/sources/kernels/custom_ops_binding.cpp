#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Launcher declaration matching custom_ops.cu
extern "C" void hardtanh_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int ndim,
    const int64_t* shape,
    const int64_t* strides,
    float min_val,
    float max_val,
    cudaStream_t stream);

// PyTorch tensor wrapper for Hardtanh forward
torch::Tensor hardtanh_forward(
    torch::Tensor x,
    double min_val,
    double max_val)
{
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must have dtype float32");
    TORCH_CHECK(x.dim() <= 4, "Input tensor must have at most 4 dimensions");

    // Allocate output with same shape, strides, dtype, device
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Extract shape and strides as int64 vectors
    std::vector<int64_t> shape_vec(x.dim());
    std::vector<int64_t> strides_vec(x.dim());
    for (int i = 0; i < x.dim(); ++i) {
        shape_vec[i] = x.size(i);
        strides_vec[i] = x.stride(i);
    }

    // Call launcher
    hardtanh_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        x.dim(),
        shape_vec.data(),
        strides_vec.data(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream
    );

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "hardtanh_launcher failed");

    return output;
}

// Registration function
void register_hardtanh_forward(pybind11::module& m) {
    m.def(
        "hardtanh_forward",
        &hardtanh_forward,
        "Hardtanh (clamp) forward",
        pybind11::arg("x"),
        pybind11::arg("min_val") = -1.0,
        pybind11::arg("max_val") = 1.0
    );
}

REGISTER_BINDING(hardtanh_forward, register_hardtanh_forward);
