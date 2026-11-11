#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void triu_forward_launcher(const float* input,
                                     float* output,
                                     int N, int C, int H, int W,
                                     cudaStream_t stream);

extern "C" void batch_norm_eval_forward_launcher(const float* input,
                                                 float* output,
                                                 const float* running_mean,
                                                 const float* running_var,
                                                 const float* weight,
                                                 const float* bias,
                                                 float eps,
                                                 int N, int C, int H, int W,
                                                 cudaStream_t stream);

// Helper to check tensor is float32 CUDA dense and contiguous
torch::Tensor check_cuda_float_contiguous(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    return t;
}

torch::Tensor triu_forward(torch::Tensor x) {
    // Validate input
    x = check_cuda_float_contiguous(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in triu_forward: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor batch_norm_eval_forward(torch::Tensor x,
                                      torch::Tensor running_mean,
                                      torch::Tensor running_var,
                                      torch::Tensor weight,
                                      torch::Tensor bias,
                                      double eps) {
    // Validate inputs
    x = check_cuda_float_contiguous(x, "x");
    running_mean = check_cuda_float_contiguous(running_mean, "running_mean");
    running_var = check_cuda_float_contiguous(running_var, "running_var");
    if (weight.defined()) {
        weight = check_cuda_float_contiguous(weight, "weight");
    }
    if (bias.defined()) {
        bias = check_cuda_float_contiguous(bias, "bias");
    }

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Check parameter shapes: should be [C]
    TORCH_CHECK(running_mean.numel() == C, "running_mean size mismatch");
    TORCH_CHECK(running_var.numel() == C, "running_var size mismatch");
    if (weight.defined()) {
        TORCH_CHECK(weight.numel() == C, "weight size mismatch");
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.numel() == C, "bias size mismatch");
    }

    auto output = torch::empty_like(x);

    const float* weight_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    batch_norm_eval_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        static_cast<float>(eps),
        N, C, H, W,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in batch_norm_eval_forward: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("triu_forward", &triu_forward,
          "Apply upper triangular mask to last two dimensions (N,C,H,W)",
          pybind11::arg("x"));
    m.def("batch_norm_eval_forward", &batch_norm_eval_forward,
          "BatchNorm2d evaluation forward",
          pybind11::arg("x"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps") = 1e-05);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
