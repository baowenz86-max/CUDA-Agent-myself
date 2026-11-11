#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linear_forward_launcher(const float* input,
                                          const float* weight,
                                          const float* bias,
                                          float* output,
                                          int64_t M,
                                          int64_t K,
                                          int64_t N,
                                          cudaStream_t stream);

extern "C" void softplus_forward_launcher(const float* input,
                                           float* output,
                                           int64_t numel,
                                           cudaStream_t stream);

extern "C" void sigmoid_forward_launcher(const float* input,
                                          float* output,
                                          int64_t numel,
                                          cudaStream_t stream);

namespace {

// Helper to ensure tensor is CUDA float32 and contiguous
void check_tensor(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    // Validate input tensors
    check_tensor(input, "input");
    check_tensor(weight, "weight");
    check_tensor(bias, "bias");

    // Input shape: [batch, seq_len, in_features]
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int64_t batch = input.size(0);
    int64_t seq_len = input.size(1);
    int64_t in_features = input.size(2);
    int64_t out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "weight dim 1 must match input dim 2");
    TORCH_CHECK(bias.size(0) == out_features, "bias size must match out_features");

    int64_t M = batch * seq_len;
    int64_t K = in_features;
    int64_t N = out_features;

    // Allocate output tensor [batch, seq_len, out_features]
    auto output = torch::empty({batch, seq_len, out_features}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    linear_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        M,
        K,
        N,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in linear_forward: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor softplus_forward(torch::Tensor input) {
    check_tensor(input, "input");

    auto output = torch::empty_like(input);

    int64_t numel = input.numel();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    softplus_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in softplus_forward: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor sigmoid_forward(torch::Tensor input) {
    check_tensor(input, "input");

    auto output = torch::empty_like(input);

    int64_t numel = input.numel();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    sigmoid_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in sigmoid_forward: ", cudaGetErrorString(err));

    return output;
}

} // anonymous namespace

void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward (input @ weight.T + bias)",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("softplus_forward", &softplus_forward,
          "Softplus activation forward",
          py::arg("input"));
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid activation forward",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
