#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_mean_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    cudaStream_t stream);

extern "C" void custom_hardswish_launcher(
    const float* input,
    float* output,
    int total_elements,
    cudaStream_t stream);

// PyTorch binding for custom_mean
torch::Tensor custom_mean_forward(torch::Tensor x, int64_t dim) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "custom_mean: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_mean: input must be float32");
    TORCH_CHECK(dim == 0, "custom_mean: only dim=0 is supported");
    TORCH_CHECK(x.dim() == 4, "custom_mean: input must be 4D");

    // Make input contiguous to match launcher's assumed memory layout
    auto x_contig = x.contiguous();

    // Extract dimensions
    int64_t batch = x_contig.size(0);
    int64_t channels = x_contig.size(1);
    int64_t height = x_contig.size(2);
    int64_t width = x_contig.size(3);

    // Allocate output tensor: shape [channels, height, width]
    auto output = torch::empty({channels, height, width}, x_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_mean_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(channels),
        static_cast<int>(height),
        static_cast<int>(width),
        stream
    );

    return output;
}

// PyTorch binding for custom_hardswish
torch::Tensor custom_hardswish_forward(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "custom_hardswish: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_hardswish: input must be float32");

    // Make input contiguous to match launcher's elementwise access
    auto x_contig = x.contiguous();

    // Allocate output tensor with same shape
    auto output = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_hardswish_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(x_contig.numel()),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_mean", &custom_mean_forward,
          "Compute mean over dim=0",
          pybind11::arg("x"),
          pybind11::arg("dim"));
    m.def("custom_hardswish", &custom_hardswish_forward,
          "Apply Hardswish activation",
          pybind11::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
