// kernels/max_pool2d_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int padH, int padW,
    int dilationH, int dilationW,
    bool ceil_mode,
    long long inputStrideN, long long inputStrideC, long long inputStrideH, long long inputStrideW,
    long long outputStrideN, long long outputStrideC, long long outputStrideH, long long outputStrideW,
    cudaStream_t stream
);

torch::Tensor max_pool2d(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode
) {
    // Validate input tensor
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    // Validate scalar parameters
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int kernelH = (int)kernel_size;
    int kernelW = (int)kernel_size;
    int strideH = (int)stride;
    int strideW = (int)stride;
    int padH = (int)padding;
    int padW = (int)padding;
    int dilationH = (int)dilation;
    int dilationW = (int)dilation;

    // Compute output spatial dimensions (mirror launcher formula)
    int outH, outW;
    if (ceil_mode) {
        outH = (H + 2 * padH - dilationH * (kernelH - 1) - 1 + strideH - 1) / strideH + 1;
        outW = (W + 2 * padW - dilationW * (kernelW - 1) - 1 + strideW - 1) / strideW + 1;
        outH = (outH < 1) ? 1 : outH;
        outW = (outW < 1) ? 1 : outW;
    } else {
        outH = (H + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1;
        outW = (W + 2 * padW - dilationW * (kernelW - 1) - 1) / strideW + 1;
    }
    TORCH_CHECK(outH > 0 && outW > 0, "output spatial dimensions must be positive");

    // Allocate output tensor with same dtype and device, contiguous
    auto output = torch::empty({N, C, outH, outW}, input.options());

    // Get input strides (may be non-contiguous)
    long long inputStrideN = input.stride(0);
    long long inputStrideC = input.stride(1);
    long long inputStrideH = input.stride(2);
    long long inputStrideW = input.stride(3);

    // Output is contiguous, so standard strides
    long long outputStrideN = (long long)C * outH * outW;
    long long outputStrideC = (long long)outH * outW;
    long long outputStrideH = outW;
    long long outputStrideW = 1;

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch CUDA kernel
    max_pool2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW,
        dilationH, dilationW,
        ceil_mode,
        inputStrideN, inputStrideC, inputStrideH, inputStrideW,
        outputStrideN, outputStrideC, outputStrideH, outputStrideW,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_max_pool2d(pybind11::module& m) {
    m.def("max_pool2d", &max_pool2d,
          "2D max pooling",
          py::arg("input"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("dilation"),
          py::arg("ceil_mode"));
}

REGISTER_BINDING(max_pool2d, register_max_pool2d);
