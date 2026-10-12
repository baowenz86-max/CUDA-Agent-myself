#include <torch/types.h>
#include <torch/library.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int output_padding_h, int output_padding_w,
    cudaStream_t stream
);

extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    float min_val, float max_val,
    int total,
    cudaStream_t stream
);

extern "C" void leaky_relu_forward_launcher(
    const float* input,
    float* output,
    float slope,
    int total,
    cudaStream_t stream
);

extern "C" void triu_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int diagonal,
    cudaStream_t stream
);

extern "C" void add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream
);

// Helper to get current CUDA stream as cudaStream_t
static inline cudaStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Wrapper functions
static torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_h,
    int64_t output_padding_w
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_in, C_out, kH, kW)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (C_out)");

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);

    // Compute output spatial dimensions using PyTorch formula
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    TORCH_CHECK(bias.size(0) == C_out, "bias size must match C_out");

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    conv_transpose2d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        output_padding_h, output_padding_w,
        get_stream()
    );

    return output;
}

static torch::Tensor clamp_forward(
    torch::Tensor input,
    double min_val,
    double max_val
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int total = input.numel();

    clamp_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        total,
        get_stream()
    );

    return output;
}

static torch::Tensor leaky_relu_forward(
    torch::Tensor input,
    double negative_slope
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int total = input.numel();

    leaky_relu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(negative_slope),
        total,
        get_stream()
    );

    return output;
}

static torch::Tensor triu_forward(
    torch::Tensor input,
    int64_t diagonal
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    int N = 1;
    int C = 1;
    if (input.dim() == 4) {
        N = input.size(0);
        C = input.size(1);
        int H = input.size(2);
        int W = input.size(3);
        auto output = torch::empty_like(input);
        triu_forward_launcher(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            N, C, H, W,
            static_cast<int>(diagonal),
            get_stream()
        );
        return output;
    } else {
        TORCH_CHECK(false, "triu_forward currently supports only 4D tensors");
    }
}

static torch::Tensor add_forward(
    torch::Tensor input,
    torch::Tensor bias
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D (C, 1, 1)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    TORCH_CHECK(bias.size(0) == C, "bias size must match channels");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must have shape (C, 1, 1)");

    auto output = torch::empty_like(input);

    add_forward_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        get_stream()
    );

    return output;
}

// Register with torch.ops.cuda_extension
TORCH_LIBRARY(cuda_extension, m) {
    m.def("conv_transpose2d_forward(Tensor input, Tensor weight, Tensor bias, int stride_h, int stride_w, int padding_h, int padding_w, int output_padding_h, int output_padding_w) -> Tensor",
          &conv_transpose2d_forward);
    m.def("clamp_forward(Tensor input, float min_val, float max_val) -> Tensor",
          &clamp_forward);
    m.def("leaky_relu_forward(Tensor input, float negative_slope) -> Tensor",
          &leaky_relu_forward);
    m.def("triu_forward(Tensor input, int diagonal) -> Tensor",
          &triu_forward);
    m.def("add_forward(Tensor input, Tensor bias) -> Tensor",
          &add_forward);
}

// Also register pybind functions for direct module access
static void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward, "ConvTranspose2d forward");
    m.def("clamp_forward", &clamp_forward, "Clamp forward");
    m.def("leaky_relu_forward", &leaky_relu_forward, "LeakyReLU forward");
    m.def("triu_forward", &triu_forward, "Triu forward");
    m.def("add_forward", &add_forward, "Add bias forward");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
