// Custom ops binding for upsample_bilinear2d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from .cu file
extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t OH, int64_t OW,
    int64_t isN, int64_t isC, int64_t isH, int64_t isW,
    int64_t osN, int64_t osC, int64_t osH, int64_t osW,
    double scale_factor,
    cudaStream_t stream);

// PyTorch wrapper
at::Tensor upsample_bilinear2d_forward(
    const at::Tensor& input,
    double scale_factor,
    bool align_corners) {
    
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "Input must be float32");
    TORCH_CHECK(scale_factor > 0, "scale_factor must be positive");
    TORCH_CHECK(!align_corners, "align_corners is not supported (must be false)");
    
    // Shape checks
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N, C, H, W)");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    
    // Compute output spatial dimensions: floor(input_size * scale_factor)
    int64_t OH = static_cast<int64_t>(std::floor(H * scale_factor));
    int64_t OW = static_cast<int64_t>(std::floor(W * scale_factor));
    TORCH_CHECK(OH > 0 && OW > 0, "Invalid output size");
    
    // Allocate output tensor (contiguous, same device, same dtype)
    auto output = at::empty({N, C, OH, OW}, input.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher
    upsample_bilinear2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        OH, OW,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale_factor,
        stream);
    
    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, 
                "CUDA kernel launch failed for upsample_bilinear2d");
    
    return output;
}

// Registration
void register_upsample_bilinear2d(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d_forward,
          "Bilinear 2D upsampling (align_corners=False)",
          pybind11::arg("input"),
          pybind11::arg("scale_factor"),
          pybind11::arg("align_corners"));
}

REGISTER_BINDING(upsample_bilinear2d, register_upsample_bilinear2d);
