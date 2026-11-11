#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_selu_forward_launcher(
    float* output,
    const float* input,
    int numel,
    int shape0, int shape1, int shape2,
    long long stride0_in, long long stride1_in, long long stride2_in,
    long long stride0_out, long long stride1_out, long long stride2_out,
    cudaStream_t stream);

extern "C" void custom_identity_forward_launcher(
    float* output,
    const float* input,
    int numel,
    int shape0, int shape1, int shape2,
    long long stride0_in, long long stride1_in, long long stride2_in,
    long long stride0_out, long long stride1_out, long long stride2_out,
    cudaStream_t stream);

// Helper to validate input tensor and return contiguous version if needed
static torch::Tensor check_float_cuda_3d(const torch::Tensor& x, const char* op_name) {
    TORCH_CHECK(x.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, op_name, ": input must be float32");
    TORCH_CHECK(x.dim() == 3, op_name, ": input must have 3 dimensions");
    return x.is_contiguous() ? x : x.contiguous();
}

torch::Tensor custom_selu_forward(torch::Tensor x) {
    x = check_float_cuda_3d(x, "custom_selu_forward");
    auto output = torch::empty_like(x);

    int shape0 = static_cast<int>(x.size(0));
    int shape1 = static_cast<int>(x.size(1));
    int shape2 = static_cast<int>(x.size(2));
    long long stride0_in = x.stride(0);
    long long stride1_in = x.stride(1);
    long long stride2_in = x.stride(2);
    long long stride0_out = output.stride(0);
    long long stride1_out = output.stride(1);
    long long stride2_out = output.stride(2);
    int numel = static_cast<int>(x.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_selu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        shape0, shape1, shape2,
        stride0_in, stride1_in, stride2_in,
        stride0_out, stride1_out, stride2_out,
        stream);
    return output;
}

torch::Tensor custom_identity_forward(torch::Tensor x) {
    x = check_float_cuda_3d(x, "custom_identity_forward");
    auto output = torch::empty_like(x);

    int shape0 = static_cast<int>(x.size(0));
    int shape1 = static_cast<int>(x.size(1));
    int shape2 = static_cast<int>(x.size(2));
    long long stride0_in = x.stride(0);
    long long stride1_in = x.stride(1);
    long long stride2_in = x.stride(2);
    long long stride0_out = output.stride(0);
    long long stride1_out = output.stride(1);
    long long stride2_out = output.stride(2);
    int numel = static_cast<int>(x.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_identity_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        numel,
        shape0, shape1, shape2,
        stride0_in, stride1_in, stride2_in,
        stride0_out, stride1_out, stride2_out,
        stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_selu_forward", &custom_selu_forward,
          "Apply SELU activation elementwise",
          py::arg("x"));
    m.def("custom_identity_forward", &custom_identity_forward,
          "Identity copy (einsum bwh->bwh)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
