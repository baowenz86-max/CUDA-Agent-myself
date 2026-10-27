// PyTorch bindings for custom_unfold and custom_max_pool1d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_unfold_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t in_channels,
    int64_t length,
    int kernel_h,
    int kernel_w,
    int dilation_h,
    int dilation_w,
    int padding_h,
    int padding_w,
    int stride_h,
    int stride_w,
    cudaStream_t stream
);

extern "C" void custom_max_pool1d_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t L,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream
);

// Check CUDA errors helper
static inline void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// custom_unfold wrapper
torch::Tensor custom_unfold_forward(
    torch::Tensor input,
    int64_t kernel_size_h,
    int64_t kernel_size_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t stride_h,
    int64_t stride_w
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "custom_unfold: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_unfold: input must be float32");
    TORCH_CHECK(input.dim() == 3, "custom_unfold: input must be 3D [batch, channels, length]");
    TORCH_CHECK(input.is_contiguous(), "custom_unfold: input must be contiguous");

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t length = input.size(2);

    // For this specific task, kernel_h=2, kernel_w=3, dilation=1, padding=0, stride=1
    // Output length computation based on reference: L_out = (in_channels - kernel_h + 1) * (length - kernel_w + 1)
    // But to be general, compute as product of spatial dims when input is viewed as [batch, 1, in_channels, length]
    int64_t out_h = (in_channels + 2 * padding_h - dilation_h * (kernel_size_h - 1) - 1) / stride_h + 1;
    int64_t out_w = (length + 2 * padding_w - dilation_w * (kernel_size_w - 1) - 1) / stride_w + 1;
    int64_t L_out = out_h * out_w;
    int64_t out_batch = batch * kernel_size_h * kernel_size_w; // channels per batch

    // Allocate output tensor (2D as per reference)
    auto output = torch::empty({out_batch, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_unfold_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        length,
        static_cast<int>(kernel_size_h),
        static_cast<int>(kernel_size_w),
        static_cast<int>(dilation_h),
        static_cast<int>(dilation_w),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        stream
    );

    check_cuda_error(cudaGetLastError(), "custom_unfold_launcher failed");
    return output;
}

// custom_max_pool1d wrapper
torch::Tensor custom_max_pool1d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "custom_max_pool1d: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_max_pool1d: input must be float32");
    TORCH_CHECK(input.dim() == 2 || input.dim() == 3, "custom_max_pool1d: input must be 2D [N, L] or 3D [N, C, L]");
    TORCH_CHECK(input.is_contiguous(), "custom_max_pool1d: input must be contiguous");

    int64_t N, C, L;
    bool squeeze_channel = false;
    if (input.dim() == 2) {
        // Treat as (N, L) with C=1
        N = input.size(0);
        C = 1;
        L = input.size(1);
        squeeze_channel = true;
    } else {
        N = input.size(0);
        C = input.size(1);
        L = input.size(2);
        squeeze_channel = false;
    }

    // Output length formula
    int64_t L_out;
    if (ceil_mode) {
        L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1 + stride - 1) / stride + 1;
    } else {
        L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    }

    // Allocate output tensor
    auto output = torch::empty({N, C, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_max_pool1d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        L,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        ceil_mode,
        stream
    );

    check_cuda_error(cudaGetLastError(), "custom_max_pool1d_launcher failed");

    if (squeeze_channel) {
        // Squeeze the singleton channel dimension to produce 2D output as reference
        return output.squeeze(1);
    }
    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("custom_unfold", &custom_unfold_forward,
          "Custom unfold operation",
          py::arg("input"),
          py::arg("kernel_size_h"),
          py::arg("kernel_size_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("padding_h"),
          py::arg("padding_w"),
          py::arg("stride_h"),
          py::arg("stride_w"));

    m.def("custom_max_pool1d", &custom_max_pool1d_forward,
          "Custom max pool1d operation",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("ceil_mode"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
