// Auto-generated PyTorch binding for custom_ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm2d_inference_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    int64_t N,
    int64_t C,
    int64_t HW,
    float eps,
    cudaStream_t stream
);

extern "C" void clamp_min_launcher(
    const float* x,
    float* y,
    float min_val,
    int64_t numel,
    cudaStream_t stream
);

// Wrapper for batch_norm2d_inference
torch::Tensor batch_norm2d_inference(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() >= 2, "x must be at least 2D (N,C,...)");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    auto C = x.size(1);
    int64_t HW = 1;
    for (int64_t i = 2; i < x.dim(); ++i) {
        HW *= x.size(i);
    }
    TORCH_CHECK(running_mean.numel() == C, "running_mean size must equal channels");
    TORCH_CHECK(running_var.numel() == C, "running_var size must equal channels");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value()) {
        auto w = weight.value();
        TORCH_CHECK(w.is_cuda(), "weight must be CUDA");
        TORCH_CHECK(w.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(w.numel() == C, "weight size must equal channels");
        weight_ptr = w.data_ptr<float>();
    }
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.numel() == C, "bias size must equal channels");
        bias_ptr = b.data_ptr<float>();
    }

    auto y = torch::empty_like(x);
    int64_t N = x.size(0);
    int64_t total = x.numel();
    if (total == 0) {
        return y;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_norm2d_inference_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        N,
        C,
        HW,
        static_cast<float>(eps),
        stream
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

// Wrapper for clamp_min
torch::Tensor clamp_min(
    torch::Tensor x,
    double min_val
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto y = torch::empty_like(x);
    int64_t numel = x.numel();
    if (numel == 0) {
        return y;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_min_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<float>(min_val),
        numel,
        stream
    );
    C10_CUDA_CHECK(cudaGetLastError());
    return y;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm2d_inference", &batch_norm2d_inference,
          "BatchNorm2d inference forward",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));
    m.def("clamp_min", &clamp_min,
          "Clamp minimum forward",
          py::arg("x"),
          py::arg("min_val"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
