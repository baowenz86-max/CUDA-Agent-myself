#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void maxpool2d_launcher(const float* input, float* output,
                                   int N, int C, int H, int W,
                                   cudaStream_t stream);

extern "C" void leaky_relu_launcher(const float* input, float* output,
                                    float negative_slope, int total,
                                    cudaStream_t stream);

// PyTorch wrapper for maxpool2d
torch::Tensor maxpool2d_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "H and W must be even for 2x2 maxpool with stride 2");

    // Output dimensions halved
    int H_out = H / 2;
    int W_out = W / 2;

    // Allocate output tensor with same dtype/device as input
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Ensure input is contiguous for simple pointer access; if not, make contiguous
    torch::Tensor x_contig = x.contiguous();

    maxpool2d_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

// PyTorch wrapper for leaky_relu
torch::Tensor leaky_relu_forward(torch::Tensor x, double negative_slope = 0.01) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Allocate output
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Ensure input is contiguous
    torch::Tensor x_contig = x.contiguous();

    leaky_relu_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(negative_slope),
        x.numel(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("maxpool2d_forward", &maxpool2d_forward,
          "2x2 max pooling with stride 2, no padding",
          py::arg("x"));
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU activation",
          py::arg("x"),
          py::arg("negative_slope") = 0.01);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
