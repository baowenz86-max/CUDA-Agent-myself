#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm_eval_2d_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream);

extern "C" void triplet_margin_loss_sum_atomic_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float margin,
    int p,
    int N,
    int D,
    float* output,
    cudaStream_t stream);

static inline void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor batch_norm_eval_2d(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps)
{
    check_cuda_float32_contiguous(x, "x");
    check_cuda_float32_contiguous(running_mean, "running_mean");
    check_cuda_float32_contiguous(running_var, "running_var");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    TORCH_CHECK(running_mean.numel() == C, "running_mean length must match C");
    TORCH_CHECK(running_var.numel() == C, "running_var length must match C");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        auto w = weight.value();
        check_cuda_float32_contiguous(w, "weight");
        TORCH_CHECK(w.numel() == C, "weight length must match C");
        weight_ptr = w.data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        auto b = bias.value();
        check_cuda_float32_contiguous(b, "bias");
        TORCH_CHECK(b.numel() == C, "bias length must match C");
        bias_ptr = b.data_ptr<float>();
    }

    auto y = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_eval_2d_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "batch_norm_eval_2d kernel failed: ", cudaGetErrorString(err));

    return y;
}

torch::Tensor triplet_margin_loss_sum(
    torch::Tensor anchor,
    torch::Tensor positive,
    torch::Tensor negative,
    double margin,
    int64_t p)
{
    check_cuda_float32_contiguous(anchor, "anchor");
    check_cuda_float32_contiguous(positive, "positive");
    check_cuda_float32_contiguous(negative, "negative");

    TORCH_CHECK(anchor.sizes() == positive.sizes(), "anchor and positive must have same shape");
    TORCH_CHECK(anchor.sizes() == negative.sizes(), "anchor and negative must have same shape");

    TORCH_CHECK(anchor.dim() >= 1, "input must have at least one dimension");
    int64_t N = anchor.size(0);
    int64_t D = anchor.numel() / N;

    auto loss = torch::empty({}, anchor.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    triplet_margin_loss_sum_atomic_launcher(
        anchor.data_ptr<float>(),
        positive.data_ptr<float>(),
        negative.data_ptr<float>(),
        static_cast<float>(margin),
        static_cast<int>(p),
        static_cast<int>(N),
        static_cast<int>(D),
        loss.data_ptr<float>(),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "triplet_margin_loss_sum kernel failed: ", cudaGetErrorString(err));

    return loss;
}

void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_eval_2d", &batch_norm_eval_2d,
          "BatchNorm2d evaluation-mode forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
    m.def("triplet_margin_loss_sum", &triplet_margin_loss_sum,
          "Triplet margin loss (sum reduction)",
          py::arg("anchor"), py::arg("positive"), py::arg("negative"),
          py::arg("margin"), py::arg("p"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
