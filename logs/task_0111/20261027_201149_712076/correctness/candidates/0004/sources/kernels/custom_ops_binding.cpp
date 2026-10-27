#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declaration of launchers from custom_ops.cu
extern "C" void ge_forward_launcher(const float* input,
                                    bool* output,
                                    int64_t numel,
                                    float other,
                                    cudaStream_t stream);

extern "C" void conv_transpose1d_launcher(const float* input,
                                         const float* weight,
                                         float* output,
                                         int batch,
                                         int in_channels,
                                         int out_channels,
                                         int input_length,
                                         int kernel_size,
                                         int stride,
                                         int padding,
                                         int output_padding,
                                         cudaStream_t stream);

// Python binding functions

torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0 && output_padding >= 0,
                "hyperparameters must be non-negative integers");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(1);
    int weight_kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight shape mismatch");
    TORCH_CHECK(weight_kernel_size == kernel_size, "kernel_size mismatch");

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "invalid output length");

    // Allocate output
    auto output = torch::empty({batch, out_channels, output_length}, input.options());

    // Ensure contiguity: copy if needed
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_channels, out_channels,
        input_length, static_cast<int>(kernel_size), static_cast<int>(stride),
        static_cast<int>(padding), static_cast<int>(output_padding),
        stream);

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in conv_transpose1d_launcher: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor ge_forward(torch::Tensor input, double other) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    ge_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<bool>(),
        input.numel(),
        static_cast<float>(other),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in ge_forward_launcher: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
    m.def("ge_forward", &ge_forward,
          "Element-wise greater than or equal",
          pybind11::arg("input"),
          pybind11::arg("other"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
