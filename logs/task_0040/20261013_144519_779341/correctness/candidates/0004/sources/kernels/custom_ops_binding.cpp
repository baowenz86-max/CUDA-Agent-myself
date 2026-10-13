// PyTorch binding for custom CUDA kernels
// All tensor validation, allocation, and launcher calls.
// ConvTranspose1d and Linear are implemented using cuDNN/cuBLAS in separate C++ files,
// but their launchers are declared here and called.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void add_bias_launcher(const float* input,
                                  const float* bias,
                                  float* output,
                                  int B, int C, int L,
                                  cudaStream_t stream);

extern "C" void roll_launcher(const float* input,
                              float* output,
                              int B, int C, int L,
                              int shifts, int dim,
                              cudaStream_t stream);

extern "C" void flatten_launcher(const float* input,
                                 float* output,
                                 int B, int C, int L,
                                 int start_dim,
                                 cudaStream_t stream);

// Forward declarations for cuDNN/cuBLAS launchers (implemented in .cu or .cpp later)
extern "C" void conv_transpose1d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_length,
    int out_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream);

extern "C" void linear_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_features,
    int out_features,
    cudaStream_t stream);

// Helper to check tensor requirements
static void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// ----------------------------------------------------------------------------
// conv_transpose1d_forward
// ----------------------------------------------------------------------------
torch::Tensor conv_transpose1d_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t output_padding) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(weight, "weight");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, in_channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [in_channels, out_channels, kernel_size]");
    TORCH_CHECK(input.size(1) == weight.size(0), "input channels must match weight in_channels");

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_length = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    // bias optional
    if (bias.defined()) {
        check_cuda_float32_contiguous(bias, "bias");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels,
                    "bias must be 1D with size out_channels");
    }

    // Compute output length
    int64_t out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "output length must be positive");

    auto output = torch::empty({batch, out_channels, out_length}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    conv_transpose1d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(in_length),
        static_cast<int>(out_length),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    return output;
}

// ----------------------------------------------------------------------------
// add_bias_forward
// ----------------------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(bias, "bias");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, channels, length]");
    TORCH_CHECK(bias.dim() == 2 && bias.size(1) == 1, "bias must be [channels, 1]");
    TORCH_CHECK(input.size(1) == bias.size(0), "bias channel dimension must match input");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_bias_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        stream
    );

    return output;
}

// ----------------------------------------------------------------------------
// roll_forward
// ----------------------------------------------------------------------------
torch::Tensor roll_forward(torch::Tensor input, int64_t shifts, int64_t dim) {
    check_cuda_float32_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, channels, length]");
    TORCH_CHECK(dim == 2, "only dim=2 is supported");
    TORCH_CHECK(shifts == 1, "only shifts=1 is supported");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    roll_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(shifts),
        static_cast<int>(dim),
        stream
    );

    return output;
}

// ----------------------------------------------------------------------------
// flatten_forward
// ----------------------------------------------------------------------------
torch::Tensor flatten_forward(torch::Tensor input, int64_t start_dim) {
    check_cuda_float32_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, channels, length]");
    TORCH_CHECK(start_dim == 1, "only start_dim=1 is supported");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    auto output = torch::empty({B, C * L}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    flatten_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(start_dim),
        stream
    );

    return output;
}

// ----------------------------------------------------------------------------
// linear_forward
// ----------------------------------------------------------------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(weight, "weight");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");
    TORCH_CHECK(input.size(1) == weight.size(1), "input in_features must match weight in_features");

    int64_t batch = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = weight.size(0);

    if (bias.defined()) {
        check_cuda_float32_contiguous(bias, "bias");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features,
                    "bias must be 1D with size out_features");
    }

    auto output = torch::empty({batch, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    linear_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        stream
    );

    return output;
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward (cuDNN)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));

    m.def("add_bias_forward", &add_bias_forward,
          "Add bias broadcast over last dimension",
          pybind11::arg("input"),
          pybind11::arg("bias"));

    m.def("roll_forward", &roll_forward,
          "Roll along last dimension by shifts",
          pybind11::arg("input"),
          pybind11::arg("shifts"),
          pybind11::arg("dim"));

    m.def("flatten_forward", &flatten_forward,
          "Flatten from start_dim to end",
          pybind11::arg("input"),
          pybind11::arg("start_dim"));

    m.def("linear_forward", &linear_forward,
          "Linear transformation (cuBLAS)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
