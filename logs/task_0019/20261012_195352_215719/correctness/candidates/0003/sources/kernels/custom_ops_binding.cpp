// PyTorch binding for custom CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int H_in, int W_in,
    int C_out, int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream
);

extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream
);

extern "C" void leaky_relu_forward_launcher(
    const float* input,
    float* output,
    int total,
    float negative_slope,
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

// Helper to get current CUDA stream
static cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// ConvTranspose2d forward
static torch::Tensor conv_transpose_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> output_padding
) {
    // Basic checks
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_in,C_out,kH,kW)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(stride.size() == 2, "stride must have 2 elements");
    TORCH_CHECK(padding.size() == 2, "padding must have 2 elements");
    TORCH_CHECK(output_padding.size() == 2, "output_padding must have 2 elements");

    int N = input.size(0);
    int C_in = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);
    int C_out = weight.size(1);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int stride_h = static_cast<int>(stride[0]);
    int stride_w = static_cast<int>(stride[1]);
    int padding_h = static_cast<int>(padding[0]);
    int padding_w = static_cast<int>(padding[1]);
    int output_padding_h = static_cast<int>(output_padding[0]);
    int output_padding_w = static_cast<int>(output_padding[1]);

    // Output dimensions formula from PyTorch ConvTranspose2d
    int H_out = (H_in - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int W_out = (W_in - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    TORCH_CHECK(C_in == weight.size(0), "weight C_in mismatch");
    TORCH_CHECK(bias.size(0) == C_out, "bias size must equal C_out");
    TORCH_CHECK(H_out > 0 && W_out > 0, "invalid output dimensions");

    // Allocate output
    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    // Call launcher
    conv_transpose_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        kernel_h, kernel_w,
        stride_h, stride_w,
        padding_h, padding_w,
        get_current_stream()
    );

    return output;
}

// Clamp forward
static torch::Tensor clamp_forward(torch::Tensor input, double min_val, double max_val) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int total = static_cast<int>(input.numel());
    clamp_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        get_current_stream()
    );
    return output;
}

// Leaky ReLU forward
static torch::Tensor leaky_relu_forward(torch::Tensor input, double negative_slope) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int total = static_cast<int>(input.numel());
    leaky_relu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        static_cast<float>(negative_slope),
        get_current_stream()
    );
    return output;
}

// Triu forward
static torch::Tensor triu_forward(torch::Tensor input, int64_t diagonal) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() >= 2, "input must have at least 2 dimensions");

    int64_t H = input.size(-2);
    int64_t W = input.size(-1);
    // Compute N*C = product of leading dimensions
    int64_t num_batches = input.numel() / (H * W);
    int N = static_cast<int>(num_batches); // actually batch*channels collapsed
    int C = 1; // not used in kernel; kernel treats N as total leading elements
    // Our kernel signature expects (N, C, H, W) but it only uses H and W for masking,
    // and N,C are only used for total = N*C*H*W. We can pass N = num_batches, C = 1.
    // However, to keep consistency, we pass N=1, C=num_batches? In kernel, total = N*C*H*W,
    // so as long as product equals numel, indexing works. We'll pass N=num_batches, C=1.
    // Wait: kernel uses total = N*C*H*W, and loops over pos < total.
    // It also computes w, h from pos, but doesn't use N or C for indexing, only for stride iteration.
    // So it's fine if N*C = num_batches. We set N = num_batches, C = 1.
    int total_elements = static_cast<int>(input.numel());
    (void) total_elements;

    auto output = torch::empty_like(input);
    triu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(num_batches),
        1,
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(diagonal),
        get_current_stream()
    );
    return output;
}

// Add bias forward (bias shape (C,1,1) broadcast over N,C,H,W)
static torch::Tensor add_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 1 && bias.size(2) == 1,
                "bias must be shape (C,1,1)");

    int N = static_cast<int>(input.size(0));
    int C = static_cast<int>(input.size(1));
    int H = static_cast<int>(input.size(2));
    int W = static_cast<int>(input.size(3));
    TORCH_CHECK(bias.size(0) == C, "bias C dimension must match input C");

    auto output = torch::empty_like(input);
    add_forward_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        get_current_stream()
    );
    return output;
}

// Registration
static void register_custom_ops(py::module& m) {
    m.def("conv_transpose_forward", &conv_transpose_forward,
          "ConvTranspose2d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("clamp_forward", &clamp_forward,
          "Clamp forward",
          py::arg("input"), py::arg("min_val"), py::arg("max_val"));
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          py::arg("input"), py::arg("negative_slope"));
    m.def("triu_forward", &triu_forward,
          "Upper triangular mask forward",
          py::arg("input"), py::arg("diagonal"));
    m.def("add_forward", &add_forward,
          "Add bias forward",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
