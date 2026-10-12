#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log10_lt_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float threshold,
    cudaStream_t stream
);

extern "C" void bias_add_launcher(
    float* output,
    const float* input,
    const float* bias,
    int64_t N,
    int64_t C,
    int64_t L,
    cudaStream_t stream
);

extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N,
    int C_in,
    int L_in,
    int C_out,
    int L_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    int groups,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// PyTorch binding functions
// -----------------------------------------------------------------------------

torch::Tensor log10_lt_forward(torch::Tensor input, double threshold) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log10_lt_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        static_cast<float>(threshold),
        stream
    );

    return output;
}

torch::Tensor bias_add_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias dtype must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");

    // Expected input shape: (N, C, L), bias shape: (C, 1) or (C,)
    TORCH_CHECK(input.dim() == 3, "Input must have 3 dimensions (N, C, L)");
    TORCH_CHECK(bias.dim() == 2 || bias.dim() == 1, "Bias must have 1 or 2 dimensions");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    if (bias.dim() == 2) {
        TORCH_CHECK(bias.size(0) == C, "Bias first dimension must match channels");
        TORCH_CHECK(bias.size(1) == 1, "Bias second dimension must be 1");
    } else {
        TORCH_CHECK(bias.size(0) == C, "Bias size must match channels");
    }

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        N,
        C,
        L,
        stream
    );

    return output;
}

torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight dtype must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias dtype must be float32");
    }
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    }

    TORCH_CHECK(input.dim() == 3, "Input must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (C_in, C_out/groups, kernel_size)");

    // Currently only support dilation == 1
    TORCH_CHECK(dilation == 1, "Only dilation=1 is supported");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1) * groups;
    int64_t kernel_size = weight.size(2);

    // Validate weight shape: (C_in, C_out/groups, kernel_size)
    TORCH_CHECK(weight.size(0) == C_in, "Weight first dimension must match input channels");
    TORCH_CHECK(weight.size(1) == C_out / groups, "Weight second dimension mismatch");

    // Validate groups
    TORCH_CHECK(C_in % groups == 0, "Input channels must be divisible by groups");
    TORCH_CHECK(C_out % groups == 0, "Output channels must be divisible by groups");

    // Compute output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    TORCH_CHECK(L_out > 0, "Output length must be positive");

    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass bias pointer: if not defined, pass nullptr
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv_transpose1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(L_in),
        static_cast<int>(C_out),
        static_cast<int>(L_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        static_cast<int>(groups),
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("log10_lt_forward", &log10_lt_forward,
          "Apply log10 and threshold (less than)",
          py::arg("input"),
          py::arg("threshold"));

    m.def("bias_add_forward", &bias_add_forward,
          "Add bias to 3D tensor (N,C,L) with bias shape (C,1) or (C,)",
          py::arg("input"),
          py::arg("bias"));

    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "1D transposed convolution (float32)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"),
          py::arg("dilation"),
          py::arg("groups"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
