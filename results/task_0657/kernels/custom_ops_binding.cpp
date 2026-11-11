#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim1_launcher(const float* input,
                                   float* output,
                                   int B, int C, int H, int W,
                                   cudaStream_t stream);

extern "C" void mean_dim2_launcher(const float* input,
                                   float* output,
                                   int B, int H, int W,
                                   cudaStream_t stream);

// Wrapper for mean over dim=1
torch::Tensor mean_dim1_forward(torch::Tensor x) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Check contiguous
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    // Check shape: must be 4D [B, C, H, W]
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B, C, H, W]");
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    // Optional: check exact dims? contract says [32,64,256,2048], but kernel generic
    // We'll keep generic but ensure C>0 etc.

    // Allocate output shape [B, 1, H, W]
    auto output = torch::empty({B, 1, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mean_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        stream
    );

    return output;
}

// Wrapper for mean over dim=2
torch::Tensor mean_dim2_forward(torch::Tensor x) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Check contiguous
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    // Check shape: must be 4D [B, 1, H, W]
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B, 1, H, W]");
    // The second dim must be 1 as per contract
    TORCH_CHECK(x.size(1) == 1, "Input dim 1 must be 1 for mean_dim2");
    int B = x.size(0);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output shape [B, 1, 1, W]
    auto output = torch::empty({B, 1, 1, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mean_dim2_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, W,
        stream
    );

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim1", &mean_dim1_forward, "Mean over dim=1 with keepdim=True",
          py::arg("x"));
    m.def("mean_dim2", &mean_dim2_forward, "Mean over dim=2 with keepdim=True",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
