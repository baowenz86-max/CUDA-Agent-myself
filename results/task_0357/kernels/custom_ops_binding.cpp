#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_maxpool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int inH, int inW,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int paddingH, int paddingW,
    int dilationH, int dilationW,
    long long in_N_stride, long long in_C_stride, long long in_H_stride, long long in_W_stride,
    cudaStream_t stream
);

extern "C" void custom_elu_launcher(
    const float* input,
    float* output,
    long long numel,
    float alpha,
    cudaStream_t stream
);

torch::Tensor custom_maxpool2d(
    torch::Tensor x,
    int64_t kernel_h, int64_t kernel_w,
    int64_t stride_h, int64_t stride_w,
    int64_t padding_h, int64_t padding_w,
    int64_t dilation_h, int64_t dilation_w,
    bool ceil_mode
) {
    TORCH_CHECK(x.is_cuda(), "custom_maxpool2d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_maxpool2d: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_maxpool2d: input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_h > 0 && kernel_w > 0, "custom_maxpool2d: kernel sizes must be positive");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "custom_maxpool2d: strides must be positive");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "custom_maxpool2d: dilations must be positive");
    // ceil_mode is not implemented in the CUDA kernel, but contract says false; we ignore it (and maybe warn in future)

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t inH = x.size(2);
    int64_t inW = x.size(3);

    int64_t outH, outW;
    if (ceil_mode) {
        outH = (inH + 2 * padding_h - dilation_h * (kernel_h - 1) - 1 + stride_h - 1) / stride_h + 1;
        outW = (inW + 2 * padding_w - dilation_w * (kernel_w - 1) - 1 + stride_w - 1) / stride_w + 1;
    } else {
        outH = (inH + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
        outW = (inW + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    }
    TORCH_CHECK(outH > 0 && outW > 0, "custom_maxpool2d: invalid output size");

    auto output = torch::empty({N, C, outH, outW}, x.options());

    // Handle strides: input might be non-contiguous; get strides in elements
    auto strides = x.strides();
    long long in_N_stride = strides[0];
    long long in_C_stride = strides[1];
    long long in_H_stride = strides[2];
    long long in_W_stride = strides[3];

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_maxpool2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)N, (int)C, (int)inH, (int)inW,
        (int)outH, (int)outW,
        (int)kernel_h, (int)kernel_w,
        (int)stride_h, (int)stride_w,
        (int)padding_h, (int)padding_w,
        (int)dilation_h, (int)dilation_w,
        in_N_stride, in_C_stride, in_H_stride, in_W_stride,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_maxpool2d: kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor custom_elu(torch::Tensor x, double alpha) {
    TORCH_CHECK(x.is_cuda(), "custom_elu: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_elu: input must be float32");

    // ELU is elementwise, output shape same as input
    auto output = torch::empty_like(x);

    long long numel = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_elu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        (float)alpha,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_elu: kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_maxpool2d", &custom_maxpool2d,
          "Custom 2D max pooling",
          py::arg("x"),
          py::arg("kernel_h") = 3,
          py::arg("kernel_w") = 5,
          py::arg("stride_h") = 2,
          py::arg("stride_w") = 2,
          py::arg("padding_h") = 0,
          py::arg("padding_w") = 0,
          py::arg("dilation_h") = 1,
          py::arg("dilation_w") = 1,
          py::arg("ceil_mode") = false);

    m.def("custom_elu", &custom_elu,
          "Custom ELU activation",
          py::arg("x"),
          py::arg("alpha") = 1.0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
