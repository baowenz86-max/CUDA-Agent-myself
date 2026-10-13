#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sigmoid_forward_launcher(const float* input,
                                         float* output,
                                         int64_t numel,
                                         cudaStream_t stream);

extern "C" void conv_transpose1d_forward_launcher(const float* input,
                                                  const float* weight,
                                                  const float* bias,
                                                  float* output,
                                                  int64_t N, int64_t C, int64_t K,
                                                  int64_t L, int64_t L_out,
                                                  int kernel_size,
                                                  int stride, int padding,
                                                  cudaStream_t stream);

// ConvTranspose1d forward
// input: [N, C, L]
// weight: [C, K, kernel_size]
// bias: [K] or undefined
// stride, padding, output_padding: ints
torch::Tensor conv_transpose1d_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t output_padding) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must have 3 dimensions (N,C,L)");
    TORCH_CHECK(weight.dim() == 3, "weight must have 3 dimensions (C,K,kernel_size)");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(1),
                    "bias must be 1D with length out_channels");
    }
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);
    int64_t K = weight.size(1);
    int64_t kernel_size = weight.size(2);
    TORCH_CHECK(weight.size(0) == C, "weight first dimension must equal in_channels");

    // Compute output length
    int64_t L_out = (L - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    auto output = torch::empty({N, K, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv_transpose1d_forward_launcher(input.data_ptr<float>(),
                                      weight.data_ptr<float>(),
                                      bias_ptr,
                                      output.data_ptr<float>(),
                                      N, C, K, L, L_out,
                                      static_cast<int>(kernel_size),
                                      static_cast<int>(stride),
                                      static_cast<int>(padding),
                                      stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Sigmoid forward
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    if (numel == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_forward_launcher(input.data_ptr<float>(),
                             output.data_ptr<float>(),
                             numel,
                             stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
