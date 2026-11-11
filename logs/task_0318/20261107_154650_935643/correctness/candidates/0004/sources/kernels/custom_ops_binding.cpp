#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm2d_forward_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void hardswish_forward_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream);

static torch::Tensor ensure_cuda_float_contiguous(torch::Tensor t, const char* name) {
    if (!t.defined()) {
        TORCH_CHECK(false, name, " must be defined");
    }
    if (t.dtype() != torch::kFloat32) {
        TORCH_CHECK(false, name, " must be float32");
    }
    if (!t.is_cuda()) {
        t = t.to(torch::kCUDA);
    }
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

torch::Tensor batch_norm2d_forward(
    torch::Tensor input,
    pybind11::object weight,
    pybind11::object bias,
    pybind11::object running_mean,
    pybind11::object running_var,
    float eps,
    float momentum,
    bool training) {

    TORCH_CHECK(input.dim() == 4, "input must be 4D [N,C,H,W]");
    input = ensure_cuda_float_contiguous(input, "input");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t total = N * C * H * W;

    TORCH_CHECK(!training, "batch_norm2d_forward only supports inference mode (training=False)");
    (void)momentum;

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    const float* running_mean_ptr = nullptr;
    const float* running_var_ptr = nullptr;

    if (!weight.is_none()) {
        torch::Tensor w = pybind11::cast<torch::Tensor>(weight);
        w = ensure_cuda_float_contiguous(w, "weight");
        TORCH_CHECK(w.numel() == C, "weight size must match channels");
        weight_ptr = w.data_ptr<float>();
    }
    if (!bias.is_none()) {
        torch::Tensor b = pybind11::cast<torch::Tensor>(bias);
        b = ensure_cuda_float_contiguous(b, "bias");
        TORCH_CHECK(b.numel() == C, "bias size must match channels");
        bias_ptr = b.data_ptr<float>();
    }
    if (running_mean.is_none()) {
        TORCH_CHECK(false, "running_mean must be provided in inference mode");
    }
    torch::Tensor rm = pybind11::cast<torch::Tensor>(running_mean);
    rm = ensure_cuda_float_contiguous(rm, "running_mean");
    TORCH_CHECK(rm.numel() == C, "running_mean size must match channels");
    running_mean_ptr = rm.data_ptr<float>();

    if (running_var.is_none()) {
        TORCH_CHECK(false, "running_var must be provided in inference mode");
    }
    torch::Tensor rv = pybind11::cast<torch::Tensor>(running_var);
    rv = ensure_cuda_float_contiguous(rv, "running_var");
    TORCH_CHECK(rv.numel() == C, "running_var size must match channels");
    running_var_ptr = rv.data_ptr<float>();

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        eps,
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_norm2d_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor hardswish_forward(torch::Tensor input) {
    TORCH_CHECK(input.dim() == 4, "input must be 4D [N,C,H,W]");
    input = ensure_cuda_float_contiguous(input, "input");
    int64_t total = input.numel();

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardswish_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(total),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hardswish_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm2d_forward", &batch_norm2d_forward,
          "BatchNorm2d inference forward",
          pybind11::arg("input"),
          pybind11::arg("weight") = pybind11::none(),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("running_mean") = pybind11::none(),
          pybind11::arg("running_var") = pybind11::none(),
          pybind11::arg("eps"),
          pybind11::arg("momentum"),
          pybind11::arg("training"));

    m.def("hardswish_forward", &hardswish_forward,
          "Hardswish forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
