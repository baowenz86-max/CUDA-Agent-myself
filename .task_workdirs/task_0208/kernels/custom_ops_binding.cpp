// Custom ops binding for BatchNorm2d inference and LogSoftmax dim1
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_batchnorm2d_forward_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void custom_log_softmax_dim1_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Helper to check CUDA errors after kernel launch
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// BatchNorm2d forward
// Signature: custom_batchnorm2d_forward(x, weight, bias, running_mean, running_var, eps) -> Tensor
torch::Tensor custom_batchnorm2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps)
{
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");

    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");

    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");

    int C = x.size(1);
    TORCH_CHECK(weight.size(0) == C, "weight size must match channels");
    TORCH_CHECK(bias.size(0) == C, "bias size must match channels");
    TORCH_CHECK(running_mean.size(0) == C, "running_mean size must match channels");
    TORCH_CHECK(running_var.size(0) == C, "running_var size must match channels");

    // Ensure contiguous tensors for raw pointer passing
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto running_mean_contig = running_mean.contiguous();
    auto running_var_contig = running_var.contiguous();

    auto y = torch::empty_like(x_contig);

    int N = x_contig.size(0);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_batchnorm2d_forward_launcher(
        x_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        running_mean_contig.data_ptr<float>(),
        running_var_contig.data_ptr<float>(),
        static_cast<float>(eps),
        y.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    check_cuda_error("custom_batchnorm2d_forward launch failed");

    return y;
}

// LogSoftmax dim=1 forward
// Signature: custom_log_softmax_dim1_forward(x) -> Tensor
torch::Tensor custom_log_softmax_dim1_forward(torch::Tensor x)
{
    // Validate input
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    auto x_contig = x.contiguous();
    auto y = torch::empty_like(x_contig);

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_log_softmax_dim1_forward_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    check_cuda_error("custom_log_softmax_dim1_forward launch failed");

    return y;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_batchnorm2d_forward", &custom_batchnorm2d_forward,
          "BatchNorm2d inference forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("eps"));
    m.def("custom_log_softmax_dim1_forward", &custom_log_softmax_dim1_forward,
          "LogSoftmax along dimension 1 (CUDA)",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
