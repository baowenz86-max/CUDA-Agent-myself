#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <vector>

// Declare launcher from custom_ops.cu
extern "C" void fused_relu_hardsigmoid_launcher(
    float* output,
    const float* input,
    long long numel,
    int shape_ndim,
    const long long* input_strides,
    const long long* output_strides,
    cudaStream_t stream);

static torch::Tensor fused_relu_hardsigmoid(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Allocate output with same shape and strides as input
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    // Prepare host arrays for strides (long long)
    int ndim = x.dim();
    std::vector<long long> input_strides(ndim);
    std::vector<long long> output_strides(ndim);
    for (int i = 0; i < ndim; ++i) {
        input_strides[i] = static_cast<long long>(x.stride(i));
        output_strides[i] = static_cast<long long>(output.stride(i));
    }

    // Pointers to strides arrays (must be accessible on device; use cudaMalloc or pinned memory)
    // Since strides are small, we can allocate device memory or use pitch.
    // For simplicity, allocate device memory for strides and copy.
    long long* d_input_strides = nullptr;
    long long* d_output_strides = nullptr;
    cudaMalloc(&d_input_strides, ndim * sizeof(long long));
    cudaMalloc(&d_output_strides, ndim * sizeof(long long));
    cudaMemcpyAsync(d_input_strides, input_strides.data(), ndim * sizeof(long long), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output_strides, output_strides.data(), ndim * sizeof(long long), cudaMemcpyHostToDevice, stream);

    // Call launcher
    fused_relu_hardsigmoid_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        ndim,
        d_input_strides,
        d_output_strides,
        stream);

    // Free device memory for strides (can be done after kernel completes, but using stream ordered free)
    cudaFree(d_input_strides);
    cudaFree(d_output_strides);

    return output;
}

static void register_fused_relu_hardsigmoid(pybind11::module& m) {
    m.def("fused_relu_hardsigmoid", &fused_relu_hardsigmoid,
          "Fused ReLU + HardSigmoid",
          pybind11::arg("x"));
}

REGISTER_BINDING(fused_relu_hardsigmoid, register_fused_relu_hardsigmoid);
