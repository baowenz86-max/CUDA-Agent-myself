#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_upsample_nearest_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    double scale_h, double scale_w,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_n, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    cudaStream_t stream);

// PyTorch wrapper function
// Input x: [N, C, H, W] CUDA tensor float32, may be non-contiguous (will use actual strides)
// scale_h, scale_w: float scaling factors
// Returns output tensor [N, C, floor(H*scale_h), floor(W*scale_w)] contiguous float32 CUDA
torch::Tensor custom_upsample_nearest(torch::Tensor x, double scale_h, double scale_w) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    // Check dimensions
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    // Check scale factors positive
    TORCH_CHECK(scale_h > 0 && scale_w > 0, "Scale factors must be positive");

    // If input is not contiguous, make it contiguous to avoid stride complexities?
    // However, the kernel supports arbitrary strides, so we can pass the actual strides.
    // To ensure correctness and avoid surprises, we will use the original tensor's strides.
    // But note: the launcher expects raw pointers, and if the tensor is non-contiguous,
    // the data_ptr works fine as long as we pass the correct strides.
    // We'll keep the tensor as is; no need to force contiguity.

    // Get input sizes and strides
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int64_t in_stride_n = x.stride(0);
    int64_t in_stride_c = x.stride(1);
    int64_t in_stride_h = x.stride(2);
    int64_t in_stride_w = x.stride(3);

    // Compute output sizes
    int outH = static_cast<int>(std::floor(H * scale_h));
    int outW = static_cast<int>(std::floor(W * scale_w));
    TORCH_CHECK(outH > 0 && outW > 0, "Output dimensions must be positive");

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Get output strides (contiguous as allocated)
    int64_t out_stride_n = output.stride(0);
    int64_t out_stride_c = output.stride(1);
    int64_t out_stride_h = output.stride(2);
    int64_t out_stride_w = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_upsample_nearest_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        scale_h, scale_w,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream);

    // Check for kernel launch errors (optional but good)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_upsample_nearest", &custom_upsample_nearest,
          "Nearest neighbor upsampling (CUDA)",
          py::arg("x"), py::arg("scale_h"), py::arg("scale_w"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
