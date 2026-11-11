#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softplus_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void conv3d_launcher(
    const float* input_ptr,
    const float* weight_ptr,
    const float* bias_ptr,
    float* output_ptr,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t K, int64_t kD, int64_t kH, int64_t kW,
    int64_t stride_d, int64_t stride_h, int64_t stride_w,
    int64_t pad_d, int64_t pad_h, int64_t pad_w,
    int64_t dilation_d, int64_t dilation_h, int64_t dilation_w,
    cudaStream_t stream
);

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation
) {
    TORCH_CHECK(input.is_cuda(), "conv3d_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv3d_forward: weight must be a CUDA tensor");
    if (bias.has_value() && bias.value().is_cuda()) {
        TORCH_CHECK(bias.value().is_cuda(), "conv3d_forward: bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv3d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv3d_forward: weight must be float32");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "conv3d_forward: bias must be float32");
    }

    // Ensure tensors are contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t D = input_contig.size(2);
    int64_t H = input_contig.size(3);
    int64_t W = input_contig.size(4);

    int64_t K = weight_contig.size(0);
    int64_t C_in = weight_contig.size(1);
    int64_t kD = weight_contig.size(2);
    int64_t kH = weight_contig.size(3);
    int64_t kW = weight_contig.size(4);

    TORCH_CHECK(C_in == C, "conv3d_forward: input channels mismatch");
    TORCH_CHECK(stride.size() == 3, "conv3d_forward: stride must have 3 elements");
    TORCH_CHECK(padding.size() == 3, "conv3d_forward: padding must have 3 elements");
    TORCH_CHECK(dilation.size() == 3, "conv3d_forward: dilation must have 3 elements");

    // Compute output dimensions
    auto calc_out = [](int64_t in_size, int64_t k, int64_t stride, int64_t pad, int64_t dilation) -> int64_t {
        return (in_size + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    };
    int64_t D_out = calc_out(D, kD, stride[0], padding[0], dilation[0]);
    int64_t H_out = calc_out(H, kH, stride[1], padding[1], dilation[1]);
    int64_t W_out = calc_out(W, kW, stride[2], padding[2], dilation[2]);

    auto output = torch::empty({N, K, D_out, H_out, W_out}, input_contig.options());

    // Prepare bias pointer
    const float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        bias_contig = bias.value().contiguous();
        TORCH_CHECK(bias_contig.size(0) == K, "conv3d_forward: bias size mismatch");
        bias_ptr = bias_contig.data_ptr<float>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv3d_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        N, C, D, H, W,
        K, kD, kH, kW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        dilation[0], dilation[1], dilation[2],
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("conv3d_launcher failed: ") + cudaGetErrorString(err));
    }

    return output;
}

torch::Tensor softplus_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "softplus_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "softplus_forward: input must be float32");
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    int64_t numel = input_contig.numel();
    if (numel == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_launcher(input_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("softplus_launcher failed: ") + cudaGetErrorString(err));
    }
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv3d_forward", &conv3d_forward,
          "3D convolution forward (cuDNN)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
    m.def("softplus_forward", &softplus_forward,
          "Softplus activation forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
