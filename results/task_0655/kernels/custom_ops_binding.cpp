#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_unfold_launcher(
    float* output,
    const float* input,
    int batch_size,
    int channels,
    int height,
    int width,
    int kernel_h,
    int kernel_w,
    int64_t batch_stride,
    int64_t channel_stride,
    int64_t height_stride,
    int64_t width_stride,
    cudaStream_t stream);

extern "C" void custom_sum_dim_launcher(
    float* output,
    const float* input,
    int64_t outer_size,
    int64_t reduce_size,
    int64_t inner_size,
    cudaStream_t stream);

// Wrapper for custom_unfold
torch::Tensor custom_unfold(torch::Tensor x, int64_t kernel_h, int64_t kernel_w) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (batch, channels, height, width)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "Kernel sizes must be positive");

    int batch_size = x.size(0);
    int channels = x.size(1);
    int height = x.size(2);
    int width = x.size(3);

    int out_h = height - kernel_h + 1;
    int out_w = width - kernel_w + 1;
    TORCH_CHECK(out_h > 0 && out_w > 0, "Kernel size too large for input spatial dimensions");

    // Output shape: (batch, channels * kernel_h * kernel_w, out_h * out_w)
    int64_t L = static_cast<int64_t>(out_h) * out_w;
    auto output = torch::empty({batch_size, static_cast<int64_t>(channels) * kernel_h * kernel_w, L}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass strides as int64_t; x.strides() returns IntArrayRef
    int64_t batch_stride = x.stride(0);
    int64_t channel_stride = x.stride(1);
    int64_t height_stride = x.stride(2);
    int64_t width_stride = x.stride(3);

    custom_unfold_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch_size,
        channels,
        height,
        width,
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        batch_stride,
        channel_stride,
        height_stride,
        width_stride,
        stream
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_unfold CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for custom_sum_dim
torch::Tensor custom_sum_dim(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "Dimension out of range");

    // Normalize negative dim
    if (dim < 0) dim += x.dim();

    // Strategy: move target dimension to last, make contiguous, and flatten to (outer, reduce, inner=1)
    // Since sum_dim_launcher expects contiguous input and reduction in the middle,
    // we can transpose to make dim the last, then contiguous, then view as (outer, reduce, 1).
    // This avoids writing a generalized strided sum kernel.
    torch::Tensor x_permuted = x;
    if (dim != x.dim() - 1) {
        x_permuted = x.transpose(dim, x.dim() - 1);
    }
    x_permuted = x_permuted.contiguous();

    int64_t reduce_size = x.size(dim);
    int64_t outer_size = x_permuted.numel() / reduce_size;  // since last dim is reduce, product of all other dims
    int64_t inner_size = 1;

    // Output shape: remove dim. We can compute expected shape from x.sizes() with dim removed.
    std::vector<int64_t> out_sizes;
    for (int64_t i = 0; i < x.dim(); ++i) {
        if (i != dim) out_sizes.push_back(x.size(i));
    }

    auto output = torch::empty(out_sizes, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with outer_size, reduce_size, inner_size=1
    // Output from launcher is a contiguous tensor of shape (outer_size, 1) = (outer_size,)
    // We pass output.data_ptr directly; launcher writes sequentially.
    custom_sum_dim_launcher(
        output.data_ptr<float>(),
        x_permuted.data_ptr<float>(),
        outer_size,
        reduce_size,
        inner_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_sum_dim CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("custom_unfold", &custom_unfold, "Custom unfold (nn.Unfold with stride=1, padding=0, dilation=1)",
          py::arg("x"), py::arg("kernel_h"), py::arg("kernel_w"));
    m.def("custom_sum_dim", &custom_sum_dim, "Custom sum along a dimension",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
