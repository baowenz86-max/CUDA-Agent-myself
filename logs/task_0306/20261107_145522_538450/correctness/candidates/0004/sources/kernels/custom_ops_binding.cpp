#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softplus_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void conv3d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t N, int64_t C_in, int64_t D, int64_t H, int64_t W,
    int64_t K, int64_t kD, int64_t kH, int64_t kW,
    const int* stride,
    const int* padding,
    const int* dilation,
    cudaStream_t stream
);

// Helper to check CUDA errors after launcher
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Conv3d forward wrapper
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
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "conv3d_forward: bias must be float32");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv3d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv3d_forward: weight must be float32");

    // Ensure contiguity for simplicity (contract allows strides but we require contiguous for launcher)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    // Extract dimensions
    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t D = input_contig.size(2);
    int64_t H = input_contig.size(3);
    int64_t W = input_contig.size(4);

    int64_t K = weight_contig.size(0);
    int64_t kD = weight_contig.size(2);
    int64_t kH = weight_contig.size(3);
    int64_t kW = weight_contig.size(4);

    // Check stride/padding/dilation lengths
    TORCH_CHECK(stride.size() == 3, "conv3d_forward: stride must have 3 elements");
    TORCH_CHECK(padding.size() == 3, "conv3d_forward: padding must have 3 elements");
    TORCH_CHECK(dilation.size() == 3, "conv3d_forward: dilation must have 3 elements");

    // Convert to int arrays
    int stride_arr[3] = {static_cast<int>(stride[0]), static_cast<int>(stride[1]), static_cast<int>(stride[2])};
    int padding_arr[3] = {static_cast<int>(padding[0]), static_cast<int>(padding[1]), static_cast<int>(padding[2])};
    int dilation_arr[3] = {static_cast<int>(dilation[0]), static_cast<int>(dilation[1]), static_cast<int>(dilation[2])};

    // Compute output dimensions
    int64_t D_out = (D + 2 * padding_arr[0] - dilation_arr[0] * (kD - 1) - 1) / stride_arr[0] + 1;
    int64_t H_out = (H + 2 * padding_arr[1] - dilation_arr[1] * (kH - 1) - 1) / stride_arr[1] + 1;
    int64_t W_out = (W + 2 * padding_arr[2] - dilation_arr[2] * (kW - 1) - 1) / stride_arr[2] + 1;

    auto output = torch::empty({N, K, D_out, H_out, W_out}, input_contig.options());

    // Get pointers
    const float* input_ptr = input_contig.data_ptr<float>();
    const float* weight_ptr = weight_contig.data_ptr<float>();
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        auto bias_contig = bias.value().contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }
    float* output_ptr = output.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv3d_launcher(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        N, C_in, D, H, W,
        K, kD, kH, kW,
        stride_arr,
        padding_arr,
        dilation_arr,
        stream
    );

    check_cuda_error("conv3d_launcher failed");

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
    check_cuda_error("softplus_launcher failed");
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
