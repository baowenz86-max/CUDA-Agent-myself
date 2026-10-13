#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void elu_forward_launcher(
    float* output,
    const float* input,
    long long numel,
    cudaStream_t stream);

extern "C" void mean_dim1_keepdim_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void add_scalar_launcher(
    float* output,
    const float* input,
    const float* scalar,
    long long numel,
    cudaStream_t stream);

static void check_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

torch::Tensor elu_forward(torch::Tensor input) {
    check_cuda_float_contiguous(input, "input");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        stream);
    return output;
}

torch::Tensor mean_dim1_keepdim_forward(torch::Tensor input) {
    check_cuda_float_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 4, "mean_dim1_keepdim_forward expects 4D tensor [N,C,H,W]");
    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    TORCH_CHECK(C > 0, "Channel dimension must be positive");

    auto output = torch::empty({N, 1, H, W}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_dim1_keepdim_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H, W,
        stream);
    return output;
}

torch::Tensor add_scalar_forward(torch::Tensor input, torch::Tensor scalar) {
    check_cuda_float_contiguous(input, "input");
    check_cuda_float_contiguous(scalar, "scalar");
    TORCH_CHECK(scalar.numel() == 1, "scalar must have exactly one element");
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_scalar_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        scalar.data_ptr<float>(),
        input.numel(),
        stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("elu_forward", &elu_forward, "ELU activation forward");
    m.def("mean_dim1_keepdim_forward", &mean_dim1_keepdim_forward, "Mean over dim=1 with keepdim=True");
    m.def("add_scalar_forward", &add_scalar_forward, "Add scalar to every element");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
