// custom_ops_binding.cpp
// PyTorch bindings for custom_stft, custom_abs, custom_remainder, custom_conv1d.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void custom_stft_launcher(
    const float* input,
    long long batch,
    long long length,
    long long n_fft,
    long long hop_length,
    float* output_real,
    float* output_imag,
    cudaStream_t stream);

extern "C" void custom_abs_launcher(
    const float* input_real,
    const float* input_imag,
    float* output,
    long long numel,
    long long stride_real_elem,
    long long stride_imag_elem,
    long long output_stride_elem,
    cudaStream_t stream);

extern "C" void custom_remainder_launcher(
    const float* input,
    float* output,
    float divisor,
    long long numel,
    long long input_stride_elem,
    long long output_stride_elem,
    cudaStream_t stream);

extern "C" void custom_conv1d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    long long batch,
    long long in_channels,
    long long length_in,
    long long out_channels,
    long long kernel_size,
    long long stride,
    long long padding,
    float* output,
    cudaStream_t stream);

// Helper: check CUDA error
static inline void check_cuda_error(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// custom_stft wrapper
// Input x: float32 CUDA tensor [batch, length]
// Output: complex64 CUDA tensor [batch, n_freq, frames] with strides [n_freq*frames, 1, n_freq]
torch::Tensor custom_stft(torch::Tensor x, int64_t n_fft, int64_t hop_length) {
    TORCH_CHECK(x.is_cuda(), "custom_stft: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_stft: input must be float32");
    TORCH_CHECK(x.dim() == 2, "custom_stft: input must be 2D");

    auto x_contig = x.contiguous();
    long long batch = x_contig.size(0);
    long long length = x_contig.size(1);
    long long n_freq = n_fft / 2 + 1;
    long long frames = 1 + length / hop_length;

    auto output = torch::empty_strided(
        {batch, n_freq, frames},
        {n_freq * frames, 1, n_freq},
        torch::dtype(torch::kComplexFloat).device(x.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get pointers to real and imag parts of output complex tensor (interleaved)
    float* output_real = reinterpret_cast<float*>(output.data_ptr<c10::complex<float>>());
    float* output_imag = output_real + 1;

    custom_stft_launcher(
        x_contig.data_ptr<float>(),
        batch,
        length,
        n_fft,
        hop_length,
        output_real,
        output_imag,
        stream);

    check_cuda_error(cudaGetLastError(), "custom_stft kernel launch failed");

    return output;
}

// custom_abs wrapper
// Input x: complex64 CUDA tensor [batch, freq, frames] (may be non-contiguous)
// Output: float32 CUDA tensor with same shape (contiguous)
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kComplexFloat, "custom_abs: input must be complex64");
    TORCH_CHECK(x.dim() == 3, "custom_abs: input must be 3D");

    // Make contiguous to simplify strides
    auto x_contig = x.contiguous();
    long long batch = x_contig.size(0);
    long long freq = x_contig.size(1);
    long long frames = x_contig.size(2);
    long long numel = x_contig.numel();

    float* input_real = reinterpret_cast<float*>(x_contig.data_ptr<c10::complex<float>>());
    float* input_imag = input_real + 1;

    auto output = torch::empty({batch, freq, frames}, torch::dtype(torch::kFloat32).device(x.device()));
    float* out_ptr = output.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // For contiguous complex, real parts are 2 floats apart, imag parts are 2 floats apart
    long long stride_real_elem = 2;
    long long stride_imag_elem = 2;
    long long output_stride_elem = 1;

    custom_abs_launcher(
        input_real,
        input_imag,
        out_ptr,
        numel,
        stride_real_elem,
        stride_imag_elem,
        output_stride_elem,
        stream);

    check_cuda_error(cudaGetLastError(), "custom_abs kernel launch failed");

    return output;
}

// custom_remainder wrapper
// Input x: float32 CUDA tensor [batch, freq, frames] (may be non-contiguous)
// Output: float32 CUDA tensor with same shape (contiguous)
torch::Tensor custom_remainder(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "custom_remainder: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_remainder: input must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_remainder: input must be 3D");

    auto x_contig = x.contiguous();
    long long numel = x_contig.numel();

    auto output = torch::empty_like(x_contig, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_remainder_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(divisor),
        numel,
        1, // input contiguous stride
        1, // output contiguous stride
        stream);

    check_cuda_error(cudaGetLastError(), "custom_remainder kernel launch failed");

    return output;
}

// custom_conv1d wrapper using cuDNN launcher
// Input x: float32 CUDA tensor [batch, in_channels, length_in] (may be non-contiguous)
// weight: float32 CUDA tensor [out_channels, in_channels, kernel_size]
// bias: float32 CUDA tensor [out_channels] (may be None)
// stride, padding: ints
// Output: float32 CUDA tensor [batch, out_channels, length_out] contiguous
torch::Tensor custom_conv1d(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "custom_conv1d: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "custom_conv1d: weight must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_conv1d: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "custom_conv1d: weight must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_conv1d: input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "custom_conv1d: weight must be 3D");

    long long batch = x.size(0);
    long long in_channels = x.size(1);
    long long length_in = x.size(2);
    long long out_channels = weight.size(0);
    long long kernel_size = weight.size(2);
    long long length_out = (length_in + 2 * padding - kernel_size) / stride + 1;

    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();
    // Bias may be None; handle later

    auto output = torch::empty({batch, out_channels, length_out}, torch::dtype(torch::kFloat32).device(x.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = nullptr;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32, "bias must be CUDA float32");
        auto b_contig = bias.contiguous();
        bias_ptr = b_contig.data_ptr<float>();
    }

    custom_conv1d_launcher(
        x_contig.data_ptr<float>(),
        w_contig.data_ptr<float>(),
        bias_ptr,
        batch,
        in_channels,
        length_in,
        out_channels,
        kernel_size,
        stride,
        padding,
        output.data_ptr<float>(),
        stream);

    check_cuda_error(cudaGetLastError(), "custom_conv1d kernel launch failed");

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_stft", &custom_stft, "Custom STFT",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"));
    m.def("custom_abs", &custom_abs, "Custom abs");
    m.def("custom_remainder", &custom_remainder, "Custom remainder",
          py::arg("x"), py::arg("divisor"));
    m.def("custom_conv1d", &custom_conv1d, "Custom conv1d using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);