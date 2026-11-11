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
    const float* input, const float* weight, const float* bias,
    float* output,
    int64_t N, int64_t C, int64_t D, int64_t H, int64_t W,
    int64_t K, int64_t kD, int64_t kH, int64_t kW,
    int64_t strideD, int64_t strideH, int64_t strideW,
    int64_t padD, int64_t padH, int64_t padW,
    int64_t dilationD, int64_t dilationH, int64_t dilationW,
    cudaStream_t stream);

static inline void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Forward function for 3D convolution
torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation
) {
    // Check device and dtype
    TORCH_CHECK(input.is_cuda(), "conv3d_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv3d_forward: weight must be a CUDA tensor");
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "conv3d_forward: bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv3d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv3d_forward: weight must be float32");
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "conv3d_forward: bias must be float32");
    }

    // Ensure input and weight are contiguous (cuDNN requires contiguous tensors)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.has_value() && bias.value().defined() ? bias.value().contiguous() : torch::Tensor();

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

    // Compute output dims
    int64_t D_out = (D + 2 * padding[0] - dilation[0] * (kD - 1) - 1) / stride[0] + 1;
    int64_t H_out = (H + 2 * padding[1] - dilation[1] * (kH - 1) - 1) / stride[1] + 1;
    int64_t W_out = (W + 2 * padding[2] - dilation[2] * (kW - 1) - 1) / stride[2] + 1;

    auto output = torch::empty({N, K, D_out, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = nullptr;
    if (bias_contig.defined()) {
        bias_ptr = bias_contig.data_ptr<float>();
    }

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

    // Check for CUDA errors after launch
    checkCudaError(cudaGetLastError(), "conv3d_launcher failed");

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
