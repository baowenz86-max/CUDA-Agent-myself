#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "../binding_registry.h"

// Declare launcher from .cu file
extern "C" void batch_norm_1d_forward_launcher(
    const float* input,
    float* output,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    int N, int C, int L,
    long long input_stride_n,
    long long input_stride_c,
    long long input_stride_l,
    long long output_stride_n,
    long long output_stride_c,
    long long output_stride_l,
    float eps,
    cudaStream_t stream
);

torch::Tensor batch_norm_1d_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Check input tensor
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, L)");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");

    // Ensure input is on CUDA
    if (!input.is_cuda()) {
        input = input.cuda();
    }

    int N = input.size(0);
    int C = input.size(1);
    int L = input.size(2);

    // Check running_mean and running_var
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(running_mean.scalar_type() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.scalar_type() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");
    TORCH_CHECK(running_mean.numel() == C, "running_mean shape mismatch");
    TORCH_CHECK(running_var.numel() == C, "running_var shape mismatch");

    // Check weight and bias if provided, ensure contiguous and on CUDA
    torch::Tensor weight_contig;
    torch::Tensor bias_contig;
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;

    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight->scalar_type() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight->is_contiguous(), "weight must be contiguous");
        TORCH_CHECK(weight->numel() == C, "weight shape mismatch");
        weight_contig = weight->contiguous();
        weight_ptr = weight_contig.data_ptr<float>();
    }

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->scalar_type() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->numel() == C, "bias shape mismatch");
        bias_contig = bias->contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    // Allocate output tensor with same shape as input (contiguous)
    auto output = torch::empty_like(input, input.options());

    // Get strides for input (element-wise, not bytes)
    auto input_strides = input.strides();
    // PyTorch strides are in elements, same as launcher expects
    long long input_stride_n = input_strides[0];
    long long input_stride_c = input_strides[1];
    long long input_stride_l = input_strides[2];

    // Output is contiguous, so its strides are C*L, L, 1
    long long output_stride_n = static_cast<long long>(C) * L;
    long long output_stride_c = L;
    long long output_stride_l = 1;

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    batch_norm_1d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        N, C, L,
        input_stride_n, input_stride_c, input_stride_l,
        output_stride_n, output_stride_c, output_stride_l,
        static_cast<float>(eps),
        stream
    );

    return output;
}

void register_batch_norm_1d(pybind11::module& m) {
    m.def("batch_norm_1d_forward", &batch_norm_1d_forward,
          "BatchNorm1d forward (eval mode)",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps") = 1e-05);
}

REGISTER_BINDING(batch_norm_1d, register_batch_norm_1d);
