#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

extern "C" {
void triu_2d_forward_launcher(
    float* output,
    const float* input,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t diagonal,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_c,
    int64_t output_stride_h,
    int64_t output_stride_w,
    cudaStream_t stream);

void batch_norm2d_eval_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t input_stride_n,
    int64_t input_stride_c,
    int64_t input_stride_h,
    int64_t input_stride_w,
    int64_t output_stride_n,
    int64_t output_stride_c,
    int64_t output_stride_h,
    int64_t output_stride_w,
    cudaStream_t stream);
}

static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

static void check_4d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dim() == 4, name, " must be 4D (N, C, H, W)");
}

static void check_1d(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.dim() == 1, name, " must be 1D");
}

torch::Tensor triu_2d_forward(torch::Tensor x, int64_t diagonal) {
    check_cuda_float32(x, "x");
    check_4d(x, "x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    triu_2d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        diagonal,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        stream
    );

    return output;
}

torch::Tensor batch_norm2d_eval_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    check_cuda_float32(input, "input");
    check_4d(input, "input");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match channels");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match channels");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;

    if (weight.has_value() && weight->defined()) {
        auto w = weight.value();
        check_cuda_float32(w, "weight");
        check_1d(w, "weight");
        TORCH_CHECK(w.is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(w.numel() == C, "weight size must match channels");
        weight_ptr = w.data_ptr<float>();
    }

    if (bias.has_value() && bias->defined()) {
        auto b = bias.value();
        check_cuda_float32(b, "bias");
        check_1d(b, "bias");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(b.numel() == C, "bias size must match channels");
        bias_ptr = b.data_ptr<float>();
    }

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm2d_eval_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N, C, H, W,
        input.stride(0), input.stride(1), input.stride(2), input.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("triu_2d_forward", &triu_2d_forward, "triu_2d_forward",
          pybind11::arg("x"), pybind11::arg("diagonal"));
    m.def("batch_norm2d_eval_forward", &batch_norm2d_eval_forward, "batch_norm2d_eval_forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("running_mean"), pybind11::arg("running_var"), pybind11::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
