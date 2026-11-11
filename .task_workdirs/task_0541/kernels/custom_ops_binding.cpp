#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_batchnorm2d_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t input_sN, int64_t input_sC, int64_t input_sH, int64_t input_sW,
    cudaStream_t stream);

extern "C" void custom_gelu_forward_launcher(
    float* output,
    const float* input,
    int64_t numel,
    int64_t input_s0, int64_t input_s1, int64_t input_s2, int64_t input_s3,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    cudaStream_t stream);

// Helper to validate common requirements
static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// PyTorch wrapper for BatchNorm2d eval
static torch::Tensor custom_batchnorm2d_eval(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    // Input validation
    check_cuda_float32(input, "input");
    check_cuda_float32(weight, "weight");
    check_cuda_float32(bias, "bias");
    check_cuda_float32(running_mean, "running_mean");
    check_cuda_float32(running_var, "running_var");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    TORCH_CHECK(weight.numel() == C, "weight size must match channel count");
    TORCH_CHECK(bias.numel() == C, "bias size must match channel count");
    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match channel count");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match channel count");

    // Ensure parameter tensors are contiguous for the kernel
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto mean_contig = running_mean.contiguous();
    auto var_contig = running_var.contiguous();

    // Allocate output with same shape as input, contiguous by default
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get input strides (in elements)
    auto strides = input.strides();
    int64_t sN = strides[0];
    int64_t sC = strides[1];
    int64_t sH = strides[2];
    int64_t sW = strides[3];

    custom_batchnorm2d_eval_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        mean_contig.data_ptr<float>(),
        var_contig.data_ptr<float>(),
        eps,
        N, C, H, W,
        sN, sC, sH, sW,
        stream);

    return output;
}

// PyTorch wrapper for GELU
static torch::Tensor custom_gelu_forward(torch::Tensor input) {
    check_cuda_float32(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    int64_t dim0 = input.size(0);
    int64_t dim1 = input.size(1);
    int64_t dim2 = input.size(2);
    int64_t dim3 = input.size(3);
    int64_t numel = input.numel();

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    auto strides = input.strides();
    int64_t s0 = strides[0];
    int64_t s1 = strides[1];
    int64_t s2 = strides[2];
    int64_t s3 = strides[3];

    custom_gelu_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        numel,
        s0, s1, s2, s3,
        dim0, dim1, dim2, dim3,
        stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_batchnorm2d_eval", &custom_batchnorm2d_eval,
          "Eval-mode BatchNorm2d",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
    m.def("custom_gelu_forward", &custom_gelu_forward,
          "GELU activation",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
