#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    cudaStream_t stream);

extern "C" void relu_forward_launcher(float* data, int64_t numel, cudaStream_t stream);

extern "C" void add_bias_forward_launcher(
    float* data,
    const float* bias,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w,
    cudaStream_t stream);

// ReflectionPad2d wrapper
torch::Tensor reflection_pad2d_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    // No need to require contiguous, kernel supports strides
    int N = input.size(0);
    int C = input.size(1);
    int H_in = input.size(2);
    int W_in = input.size(3);

    // Get padded dimensions
    int H_out = H_in + pad_top + pad_bottom;
    int W_out = W_in + pad_left + pad_right;

    // Allocate output tensor with standard contiguous strides (like PyTorch reflection_pad2d)
    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    // Get strides for input and output tensors
    auto in_strides = input.strides();
    auto out_strides = output.strides();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    reflection_pad2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        (int)pad_left, (int)pad_right, (int)pad_top, (int)pad_bottom,
        (int)in_strides[0], (int)in_strides[1], (int)in_strides[2], (int)in_strides[3],
        (int)out_strides[0], (int)out_strides[1], (int)out_strides[2], (int)out_strides[3],
        stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "reflection_pad2d_forward kernel launch failed");

    return output;
}

// ReLU wrapper (in-place)
torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous for in-place ReLU");

    int64_t numel = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(input.data_ptr<float>(), numel, stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "relu_forward kernel launch failed");

    // Return the same tensor (in-place)
    return input;
}

// Add bias wrapper (in-place)
torch::Tensor add_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.dim() == 3 && bias.size(0) == input.size(1) && bias.size(1) == 1 && bias.size(2) == 1,
        "bias must have shape [C,1,1]");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto strides = input.strides();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_bias_forward_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W,
        (int)strides[0], (int)strides[1], (int)strides[2], (int)strides[3],
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_forward kernel launch failed");

    return input;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "ReflectionPad2d forward",
          py::arg("input"), py::arg("pad_left"), py::arg("pad_right"), py::arg("pad_top"), py::arg("pad_bottom"));
    m.def("relu_forward", &relu_forward,
          "ReLU forward (in-place)",
          py::arg("input"));
    m.def("add_forward", &add_forward,
          "Add bias forward (in-place)",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
