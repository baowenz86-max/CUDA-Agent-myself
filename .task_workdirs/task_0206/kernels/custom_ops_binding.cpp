#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare CUDA launchers from custom_ops.cu
extern "C" void leaky_relu_fwd_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float negative_slope,
    cudaStream_t stream
);

extern "C" void abs_fwd_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
);

// Wrapper for leaky_relu_fwd
torch::Tensor leaky_relu_fwd(torch::Tensor input, double negative_slope = 0.01) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    // Ensure contiguous layout for the kernel (kernel assumes contiguous)
    auto input_contig = input.contiguous();

    // Allocate output with same shape and options as contiguous input
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    leaky_relu_fwd_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        input_contig.numel(),
        static_cast<float>(negative_slope),
        stream
    );

    return output;
}

// Wrapper for abs_fwd
torch::Tensor abs_fwd(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input tensor must be float32");

    // Ensure contiguous layout for the kernel
    auto input_contig = input.contiguous();

    // Allocate output with same shape and options as contiguous input
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    abs_fwd_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        input_contig.numel(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("leaky_relu_fwd", &leaky_relu_fwd, 
          "Leaky ReLU forward",
          py::arg("input"),
          py::arg("negative_slope") = 0.01);
    m.def("abs_fwd", &abs_fwd,
          "Absolute value forward",
          py::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
