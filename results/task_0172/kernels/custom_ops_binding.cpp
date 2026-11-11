// PyTorch bindings for custom ops.
// File: kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// -----------------------------------------------------------------------------
// Forward declarations for launchers defined in custom_ops.cu
// -----------------------------------------------------------------------------

extern "C" void triu_last2d_launcher(
    const float* x,
    float* output,
    int batch,
    int channels,
    int height,
    int width,
    long long stride_batch,
    long long stride_channel,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream
);

extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    float* output,
    int batch,
    int in_channels,
    int in_h,
    int in_w,
    int out_channels,
    int out_h,
    int out_w,
    int k_h,
    int k_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// Helper: check CUDA error
// -----------------------------------------------------------------------------
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// -----------------------------------------------------------------------------
// PyTorch wrapper functions
// -----------------------------------------------------------------------------

// triu_last2d
static torch::Tensor triu_last2d(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "triu_last2d input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "triu_last2d input must be float32");
    TORCH_CHECK(x.dim() >= 2, "triu_last2d input must have at least 2 dimensions");

    int batch = 1;
    int channels = 1;
    int64_t height = 1;
    int64_t width = 1;
    if (x.dim() == 2) {
        batch = 1;
        channels = 1;
        height = x.size(0);
        width = x.size(1);
    } else if (x.dim() == 3) {
        batch = 1;
        channels = x.size(0);
        height = x.size(1);
        width = x.size(2);
    } else if (x.dim() == 4) {
        batch = x.size(0);
        channels = x.size(1);
        height = x.size(2);
        width = x.size(3);
    } else {
        TORCH_CHECK(false, "triu_last2d only supports 2D, 3D, or 4D tensors");
    }

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_last2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        (int)height,
        (int)width,
        (long long)x.stride(0),
        (long long)x.stride(1),
        (long long)x.stride(2),
        (long long)x.stride(3),
        stream
    );
    checkCudaError(cudaGetLastError(), "triu_last2d_launcher");
    return output;
}

// conv2d wrapper
static torch::Tensor conv2d(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "conv2d input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv2d weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv2d input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv2d weight must be float32");
    TORCH_CHECK(input.dim() == 4, "conv2d input must be 4D (NCHW)");
    TORCH_CHECK(weight.dim() == 4, "conv2d weight must be 4D");

    // Make contiguous (required by our custom kernel and also typical for cuDNN)
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    int k_h = weight.size(2);
    int k_w = weight.size(3);
    TORCH_CHECK(weight.size(1) == in_channels, "weight has wrong in_channels");

    // Fixed convolution parameters from reference: stride=(1,1), padding=(1,2), dilation=(2,1)
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 2;
    int dilation_h = 2, dilation_w = 1;

    // Compute output dimensions
    int out_h = (in_h + 2 * pad_h - dilation_h * (k_h - 1) - 1) / stride_h + 1;
    int out_w = (in_w + 2 * pad_w - dilation_w * (k_w - 1) - 1) / stride_w + 1;

    // Create output tensor (contiguous)
    auto output = torch::empty({batch, out_channels, out_h, out_w}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        out_h,
        out_w,
        k_h,
        k_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        dilation_h,
        dilation_w,
        stream
    );
    checkCudaError(cudaGetLastError(), "conv2d_launcher");
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------

void register_custom_ops(pybind11::module& m) {
    m.def("triu_last2d", &triu_last2d, "Apply torch.triu to last two dimensions", py::arg("x"));
    m.def("conv2d", &conv2d, "2D convolution using custom CUDA kernel", py::arg("input"), py::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
