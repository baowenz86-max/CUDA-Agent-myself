#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void sum_dim_forward_launcher(
    const float* x,
    float* out,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    cudaStream_t stream);

// PyTorch wrapper for sum over dimension 1
torch::Tensor sum_dim_forward(torch::Tensor x, int64_t dim) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    // Make contiguous to match launcher indexing, if necessary
    auto x_contig = x.contiguous();

    int64_t batch = x_contig.size(0);
    int64_t channels = x_contig.size(1);
    int64_t height = x_contig.size(2);
    int64_t width = x_contig.size(3);

    // Allocate output tensor: [batch, height, width]
    auto out = torch::empty({batch, height, width}, x_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    sum_dim_forward_launcher(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        stream
    );

    return out;
}

// Registration function
void register_sum_dim_forward(pybind11::module& m) {
    m.def("sum_dim_forward", &sum_dim_forward,
          "Sum over dimension 1 (channels)",
          py::arg("x"), py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(sum_dim_forward, register_sum_dim_forward);
