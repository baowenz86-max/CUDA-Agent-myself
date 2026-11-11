#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <cstdint>

// Declare launchers from custom_ops.cu
extern "C" void avg_pool2d_forward(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t inputNStride,
    int64_t inputCStride,
    int64_t inputHStride,
    int64_t inputWStride,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool ceil_mode,
    cudaStream_t stream
);

extern "C" void leaky_relu_forward(
    const float* input,
    float* output,
    int64_t total,
    double negative_slope,
    cudaStream_t stream
);

// Helper to compute output dimensions for average pooling
static inline std::pair<int64_t, int64_t> compute_output_size(
    int64_t H,
    int64_t W,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool ceil_mode)
{
    int64_t outH, outW;
    if (ceil_mode) {
        outH = (H + 2 * padding_h - kernel_h + stride_h - 1) / stride_h + 1;
        outW = (W + 2 * padding_w - kernel_w + stride_w - 1) / stride_w + 1;
    } else {
        outH = (H + 2 * padding_h - kernel_h) / stride_h + 1;
        outW = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    }
    // Ensure at least 1 output if input non-empty
    if (outH < 1) outH = 1;
    if (outW < 1) outW = 1;
    return {outH, outW};
}

torch::Tensor avg_pool2d(
    torch::Tensor input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool ceil_mode)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "avg_pool2d: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "avg_pool2d: input must be float32");
    TORCH_CHECK(input.dim() == 4, "avg_pool2d: input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "avg_pool2d: kernel size must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "avg_pool2d: stride must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "avg_pool2d: padding must be non-negative");

    // Dimensions
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // Get strides (in elements)
    int64_t inputNStride = input.stride(0);
    int64_t inputCStride = input.stride(1);
    int64_t inputHStride = input.stride(2);
    int64_t inputWStride = input.stride(3);

    auto [outH, outW] = compute_output_size(H, W, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, ceil_mode);

    auto output = torch::empty({N, C, outH, outW}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool2d_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        inputNStride, inputCStride, inputHStride, inputWStride,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        ceil_mode,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "avg_pool2d: CUDA error: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor leaky_relu(
    torch::Tensor input,
    double negative_slope)
{
    TORCH_CHECK(input.is_cuda(), "leaky_relu: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "leaky_relu: input must be float32");

    auto output = torch::empty_like(input);

    int64_t total = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        negative_slope,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "leaky_relu: CUDA error: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("avg_pool2d", &avg_pool2d,
          "Average Pooling 2D forward",
          py::arg("input"), py::arg("kernel_h"), py::arg("kernel_w"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("padding_h"), py::arg("padding_w"), py::arg("ceil_mode"));
    m.def("leaky_relu", &leaky_relu,
          "Leaky ReLU forward",
          py::arg("input"), py::arg("negative_slope"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
