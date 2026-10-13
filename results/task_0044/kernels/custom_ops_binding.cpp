#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int stride, int padding,
    int H_out, int W_out,
    cudaStream_t stream
);
extern "C" void leaky_relu_launcher(float* output, const float* input, float negative_slope, long long total_elements, cudaStream_t stream);
extern "C" void add_bias_launcher(float* output, const float* input, const float* bias, int C, int HW, long long total_elements, cudaStream_t stream);

// conv2d_forward using custom CUDA kernel
// input: [N, C, H, W], weight: [K, C, R, S], bias: [K]
// stride, padding, dilation, groups are integers; dilation and groups must be 1 for custom kernel
// returns output [N, K, H_out, W_out]
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (K, C, R, S)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(dilation == 1, "dilation must be 1");
    TORCH_CHECK(groups == 1, "groups must be 1");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    // Make contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);
    int K = weight_contig.size(0);
    int R = weight_contig.size(2);
    int S = weight_contig.size(3);

    TORCH_CHECK(K == bias_contig.size(0), "bias size must match output channels");
    TORCH_CHECK(weight_contig.size(1) == C, "weight has incorrect input channels");

    int H_out = (H + 2 * padding - (R - 1) - 1) / stride + 1;
    int W_out = (W + 2 * padding - (S - 1) - 1) / stride + 1;

    auto output = torch::empty({N, K, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv2d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        K, R, S,
        (int)stride, (int)padding,
        H_out, W_out,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in conv2d_forward: ", cudaGetErrorString(err));
    return output;
}

// leaky_relu_forward
torch::Tensor leaky_relu_forward(torch::Tensor input, double negative_slope) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        static_cast<float>(negative_slope),
        input.numel(),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in leaky_relu_forward: ", cudaGetErrorString(err));
    return output;
}

// add_bias_forward
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D [C,1,1]");

    int C = input.size(1);
    int HW = input.size(2) * input.size(3);
    TORCH_CHECK(bias.size(0) == C, "bias channel dimension must match input channels");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must have shape [C,1,1]");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        C,
        HW,
        input.numel(),
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in add_bias_forward: ", cudaGetErrorString(err));
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward,
          "2D convolution forward using CUDA kernel",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"),
          pybind11::arg("groups"));
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          pybind11::arg("input"),
          pybind11::arg("negative_slope"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias with broadcasting",
          pybind11::arg("input"),
          pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
