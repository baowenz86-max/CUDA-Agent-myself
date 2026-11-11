#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

extern "C" void group_norm_forward_launcher(
    const float* input,
    float* output,
    const float* weight,
    const float* bias,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t num_groups,
    double eps,
    cudaStream_t stream);

torch::Tensor group_norm_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    int64_t num_groups,
    double eps) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");

    auto input_contig = input.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    TORCH_CHECK(num_groups > 0, "num_groups must be positive");
    TORCH_CHECK(C % num_groups == 0, "C must be divisible by num_groups");

    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;

    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight->dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight->numel() == C, "weight must have C elements");
        auto weight_contig = weight->contiguous();
        weight_ptr = weight_contig.data_ptr<float>();
    }

    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->numel() == C, "bias must have C elements");
        auto bias_contig = bias->contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    group_norm_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        N,
        C,
        H,
        W,
        num_groups,
        eps,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_group_norm_forward(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward,
          "GroupNorm forward",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("num_groups"),
          py::arg("eps"));
}

REGISTER_BINDING(group_norm_forward, register_group_norm_forward);
