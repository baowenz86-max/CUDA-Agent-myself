// Binding for custom_mean_dim1_keepdim
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declaration matching custom_ops.cu
extern "C" void custom_mean_dim1_keepdim_launcher(
    float* output,
    const float* input,
    int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t sxB, int64_t sxC, int64_t sxH, int64_t sxW,
    int64_t soB, int64_t so0, int64_t soH, int64_t soW,
    cudaStream_t stream);

// Wrapper: computes mean along dim=1 with keepdim=True
torch::Tensor custom_mean_dim1_keepdim(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D [B,C,H,W]");

    // Dimensions
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C > 0, "Channels must be positive");

    // Strides (possibly non-contiguous)
    int64_t sxB = x.stride(0);
    int64_t sxC = x.stride(1);
    int64_t sxH = x.stride(2);
    int64_t sxW = x.stride(3);

    // Allocate output [B, 1, H, W] contiguous
    auto output = torch::empty({B, 1, H, W}, x.options());
    int64_t soB = output.stride(0);
    int64_t so0 = output.stride(1);  // unused for indexing because dim size=1, but pass as required
    int64_t soH = output.stride(2);
    int64_t soW = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    custom_mean_dim1_keepdim_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        B, C, H, W,
        sxB, sxC, sxH, sxW,
        soB, so0, soH, soW,
        stream);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_mean_dim1_keepdim(pybind11::module& m) {
    m.def("custom_mean_dim1_keepdim", &custom_mean_dim1_keepdim,
          "Mean along dim=1 with keepdim=True",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_mean_dim1_keepdim, register_custom_mean_dim1_keepdim);
