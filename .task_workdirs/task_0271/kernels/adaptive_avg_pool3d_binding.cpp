// Binding for adaptive_avg_pool3d_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void adaptive_avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int out_d, int out_h, int out_w,
    cudaStream_t stream);

// Wrapper function matching Python call: adaptive_avg_pool3d_forward(x, output_size)
// Contract expects output_size as a single int. We use it for all three output dims.
torch::Tensor adaptive_avg_pool3d_forward(torch::Tensor x, int64_t output_size) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Check dim
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N, C, D, H, W) tensor");

    // Ensure contiguous
    x = x.contiguous();

    // Extract dimensions
    int N = x.size(0);
    int C = x.size(1);
    int D = x.size(2);
    int H = x.size(3);
    int W = x.size(4);

    // Output size
    int out_d = static_cast<int>(output_size);
    int out_h = static_cast<int>(output_size);
    int out_w = static_cast<int>(output_size);

    // Allocate output tensor
    auto output = torch::empty({N, C, out_d, out_h, out_w}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    adaptive_avg_pool3d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D, H, W,
        out_d, out_h, out_w,
        stream
    );

    return output;
}

// Registration function
void register_adaptive_avg_pool3d(pybind11::module& m) {
    m.def("adaptive_avg_pool3d_forward", &adaptive_avg_pool3d_forward,
          "Adaptive average pooling 3D forward",
          py::arg("x"), py::arg("output_size"));
}

REGISTER_BINDING(adaptive_avg_pool3d, register_adaptive_avg_pool3d);
