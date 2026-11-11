// custom_ops_binding.cpp - PyTorch bindings for threshold_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void threshold_forward_launcher(
    float* output,
    const float* input,
    float threshold,
    float value,
    long long total_elements,
    cudaStream_t stream);

// PyTorch wrapper
static torch::Tensor threshold_forward(torch::Tensor input, double threshold, double value) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.numel() > 0, "input must not be empty");

    // Make input contiguous to simplify pointer access
    auto input_contig = input.contiguous();

    // Output shape same as input, contiguous
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    threshold_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        static_cast<float>(threshold),
        static_cast<float>(value),
        static_cast<long long>(input_contig.numel()),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "threshold_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_threshold_ops(pybind11::module& m) {
    m.def("threshold_forward", &threshold_forward,
          "Threshold forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("threshold"),
          pybind11::arg("value"));
}

REGISTER_BINDING(threshold_ops, register_threshold_ops);
