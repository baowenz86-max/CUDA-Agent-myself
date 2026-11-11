// Binding for adaptive max pooling 2D
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void adaptive_max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream
);

// PyTorch wrapper function
torch::Tensor adaptive_max_pool2d_forward(torch::Tensor x, py::tuple output_size) {
    // Check input tensor
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional (N, C, H, W)");

    // Ensure contiguous for direct pointer access
    auto x_contig = x.contiguous();

    // Get dimensions
    int N = static_cast<int>(x_contig.size(0));
    int C = static_cast<int>(x_contig.size(1));
    int H = static_cast<int>(x_contig.size(2));
    int W = static_cast<int>(x_contig.size(3));

    // Parse output_size tuple (must be length 2)
    TORCH_CHECK(output_size.size() == 2, "output_size must be a tuple of (outH, outW)");
    int outH = output_size[0].cast<int>();
    int outW = output_size[1].cast<int>();
    TORCH_CHECK(outH > 0 && outW > 0, "output_size must be positive");

    // Allocate output tensor with same options (device, dtype)
    auto output = torch::empty({N, C, outH, outW}, x_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    adaptive_max_pool2d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "adaptive_max_pool2d_launcher failed with error: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_adaptive_max_pool2d(pybind11::module& m) {
    m.def("adaptive_max_pool2d_forward", &adaptive_max_pool2d_forward,
          "Adaptive max pooling 2D forward (CUDA)",
          py::arg("x"), py::arg("output_size"));
}

// Auto-register
REGISTER_BINDING(adaptive_max_pool2d, register_adaptive_max_pool2d);
