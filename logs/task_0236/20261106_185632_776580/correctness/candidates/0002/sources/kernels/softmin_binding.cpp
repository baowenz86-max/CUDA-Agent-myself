#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void softmin_dim1_launcher(const float* input, float* output,
                                      int B, int D, int H,
                                      cudaStream_t stream);

// PyTorch wrapper for softmin along dim=1
torch::Tensor softmin_dim1(torch::Tensor x) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (B, D, H)");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Allocate output tensor with same shape, dtype, device
    auto output = torch::empty_like(x);

    // Get dimensions
    int B = x.size(0);
    int D = x.size(1);
    int H = x.size(2);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call CUDA launcher
    softmin_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, D, H,
        stream
    );

    return output;
}

// Registration function
void register_softmin_binding(pybind11::module& m) {
    m.def("softmin_dim1", &softmin_dim1,
          "softmin_dim1(x) -> Tensor\n"
          "Applies softmin along dim=1 of a 3D contiguous float32 CUDA tensor.\n"
          "Arguments:\n"
          "  x (Tensor): input tensor of shape (B, D, H)\n"
          "Returns:\n"
          "  Tensor: output tensor of same shape");
}

// Auto-register
REGISTER_BINDING(softmin_binding, register_softmin_binding);
