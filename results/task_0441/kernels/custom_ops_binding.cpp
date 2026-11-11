#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batchnorm2d_eval_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c, long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_c, long long output_stride_h, long long output_stride_w,
    int has_weight, int has_bias,
    cudaStream_t stream
);

extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long input_stride_n, long long input_stride_c, long long input_stride_h, long long input_stride_w,
    long long output_stride_n, long long output_stride_c, long long output_stride_h, long long output_stride_w,
    cudaStream_t stream
);

// BatchNorm2d evaluation wrapper
torch::Tensor batchnorm2d_eval_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match channels");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match channels");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    int has_weight = 0;
    int has_bias = 0;

    if (weight.has_value()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.numel() == C, "weight size must match channels");
        weight_ptr = w.data_ptr<float>();
        has_weight = 1;
    }
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.numel() == C, "bias size must match channels");
        bias_ptr = b.data_ptr<float>();
        has_bias = 1;
    }

    // Allocate output with same options as input
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides in elements
    auto input_strides = input.strides();
    auto output_strides = output.strides();

    batchnorm2d_eval_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N, C, H, W,
        (long long)input_strides[0],
        (long long)input_strides[1],
        (long long)input_strides[2],
        (long long)input_strides[3],
        (long long)output_strides[0],
        (long long)output_strides[1],
        (long long)output_strides[2],
        (long long)output_strides[3],
        has_weight, has_bias,
        stream
    );

    // Check for errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batchnorm2d_eval_forward launcher failed: ", cudaGetErrorString(err));

    return output;
}

// ReLU wrapper
torch::Tensor relu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto input_strides = input.strides();
    auto output_strides = output.strides();

    relu_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H, W,
        (long long)input_strides[0],
        (long long)input_strides[1],
        (long long)input_strides[2],
        (long long)input_strides[3],
        (long long)output_strides[0],
        (long long)output_strides[1],
        (long long)output_strides[2],
        (long long)output_strides[3],
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu_forward launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("batchnorm2d_eval_forward", &batchnorm2d_eval_forward,
          "BatchNorm2d evaluation forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
    m.def("relu_forward", &relu_forward,
          "ReLU forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
