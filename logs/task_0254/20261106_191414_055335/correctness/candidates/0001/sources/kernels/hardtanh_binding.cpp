#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of the CUDA launcher from custom_ops.cu
extern "C" void hardtanh_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    int64_t num_dims,
    const int64_t* shape,
    const int64_t* input_strides,
    const int64_t* output_strides,
    float min_val,
    float max_val,
    cudaStream_t stream);

// Wrapper function for the Python-visible operator
torch::Tensor hardtanh_forward(torch::Tensor x, double min_val, double max_val) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Allocate output with same shape (empty_like preserves strides but we'll handle arbitrary strides)
    auto output = torch::empty_like(x);

    // Extract metadata as int64_t vectors to ensure stable pointers
    auto shape_vec = x.sizes().vec();
    auto input_strides_vec = x.strides().vec();
    auto output_strides_vec = output.strides().vec();
    int64_t numel = x.numel();
    int64_t num_dims = x.dim();

    // Call launcher
    hardtanh_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        num_dims,
        shape_vec.data(),
        input_strides_vec.data(),
        output_strides_vec.data(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_hardtanh(pybind11::module& m) {
    m.def("hardtanh_forward", &hardtanh_forward,
          "Hardtanh forward (element-wise clamp)",
          py::arg("x"),
          py::arg("min_val") = -1.0,
          py::arg("max_val") = 1.0);
}

// Auto-register
REGISTER_BINDING(hardtanh, register_hardtanh);
