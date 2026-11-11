#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_batchnorm2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    float* running_mean,
    float* running_var,
    float eps,
    bool training,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    float momentum,
    cudaStream_t stream
);

extern "C" void custom_gelu_launcher(
    float* output,
    const float* input,
    long long numel,
    cudaStream_t stream
);

// custom_batchnorm2d binding
torch::Tensor custom_batchnorm2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    bool training
) {
    // Check input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() >= 2, "input must be at least 2D");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = (input.dim() == 2) ? 1 : input.size(2);
    int64_t W = (input.dim() == 2) ? 1 : input.size(3);
    if (input.dim() == 2) {
        H = 1;
        W = input.size(1);
        N = input.size(0);
        C = 1;
    }
    // For BatchNorm2d, input is expected 4D, but handle 2D as N,C? Not required.
    // Adjust: if dim==2, treat as N,C? Actually BatchNorm2d expects 4D; we can enforce 4D.
    TORCH_CHECK(input.dim() == 4, "BatchNorm2d expects 4D input");
    N = input.size(0);
    C = input.size(1);
    H = input.size(2);
    W = input.size(3);

    // Check weight, bias optional; if not None, must be CUDA float32 and size C
    bool has_weight = weight.numel() > 0;
    bool has_bias = bias.numel() > 0;
    if (has_weight) {
        TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.numel() == C, "weight size must match C");
        TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    }
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.numel() == C, "bias size must match C");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
    // running_mean/var required
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32 && running_mean.numel() == C, "running_mean must be CUDA float32 of size C");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32 && running_var.numel() == C, "running_var must be CUDA float32 of size C");

    // Allocate output tensor (contiguous)
    auto output = torch::empty_like(input);
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    // Get strides (input may be non-contiguous; output is contiguous)
    long long in_stride_n = input.stride(0);
    long long in_stride_c = input.stride(1);
    long long in_stride_h = input.stride(2);
    long long in_stride_w = input.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // momentum hardcoded to 0.1 as in BatchNorm2d default, but not used in eval mode.
    // The launcher accepts momentum, but in eval mode it's unused.
    float momentum = 0.1f;

    // Call launcher
    custom_batchnorm2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        has_weight ? weight.data_ptr<float>() : nullptr,
        has_bias ? bias.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        training,
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        momentum,
        stream
    );

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_batchnorm2d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// custom_gelu binding
torch::Tensor custom_gelu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_gelu_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<long long>(input.numel()),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_gelu kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_batchnorm2d", &custom_batchnorm2d,
          "BatchNorm2d forward (custom CUDA)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"),
          py::arg("training"));
    m.def("custom_gelu", &custom_gelu,
          "GELU activation (custom CUDA)",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
