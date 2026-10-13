#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void ge_forward_launcher(const float* input, bool* output, int64_t numel, float threshold, cudaStream_t stream);
extern "C" void relu6_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void mul_forward_launcher(const float* input, const float* mask, float* output, int64_t numel, cudaStream_t stream);
extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int input_length,
    int kernel_size,
    int stride,
    int padding,
    int output_length,
    cudaStream_t stream);

// Helper to check CUDA errors
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// ConvTranspose1d forward implementation
static torch::Tensor conv_transpose1d_forward_impl(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    // Convert to contiguous if needed
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t input_length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(1);
    int64_t kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(0) == in_channels, "weight size[0] must match input channels");
    TORCH_CHECK(bias_contig.size(0) == out_channels, "bias size must equal out_channels");

    // Compute output length
    int64_t output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "output length must be positive");

    // Allocate output tensor
    auto output = torch::empty({batch, out_channels, output_length}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    conv_transpose1d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(input_length),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_length),
        stream
    );

    return output;
}

// Python binding functions
 torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    return conv_transpose1d_forward_impl(input, weight, bias, stride, padding, output_padding);
}

torch::Tensor ge_forward(torch::Tensor input, double threshold) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    auto input_contig = input.contiguous();
    auto output = torch::empty(input_contig.sizes(), input_contig.options().dtype(torch::kBool));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    ge_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        input_contig.numel(),
        static_cast<float>(threshold),
        stream
    );
    return output;
}

torch::Tensor relu6_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu6_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        input_contig.numel(),
        stream
    );
    return output;
}

torch::Tensor mul_forward(torch::Tensor input, torch::Tensor mask) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(mask.is_cuda(), "mask must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(mask.dtype() == torch::kFloat32, "mask must be float32");
    TORCH_CHECK(input.sizes() == mask.sizes(), "input and mask must have same shape");
    auto input_contig = input.contiguous();
    auto mask_contig = mask.contiguous();
    auto output = torch::empty_like(input_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mul_forward_launcher(
        input_contig.data_ptr<float>(),
        mask_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        input_contig.numel(),
        stream
    );
    return output;
}

void register_custom_ops_binding(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("ge_forward", &ge_forward,
          "element-wise greater than or equal",
          py::arg("input"), py::arg("threshold"));
    m.def("relu6_forward", &relu6_forward,
          "ReLU6 activation",
          py::arg("input"));
    m.def("mul_forward", &mul_forward,
          "element-wise multiplication",
          py::arg("input"), py::arg("mask"));
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
