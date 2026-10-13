// Binding for custom_add, custom_argmin, custom_stft
// This file contains only PyTorch tensor wrappers and pybind registrations.
// CUDA kernels and launchers are implemented in custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_add_launcher(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream);
extern "C" void custom_argmin_launcher(const float* x, int64_t* out, int64_t rows, int64_t row_len, cudaStream_t stream);
extern "C" void custom_stft_launcher(const float* input, float2* output, int64_t batch, int64_t audio_len, int64_t n_fft, int64_t hop_length, int64_t win_length, cudaStream_t stream);

// Helper to validate CUDA float32 contiguous tensor
static void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// custom_add: element-wise addition
torch::Tensor custom_add(torch::Tensor x, torch::Tensor y) {
    check_cuda_float32_contiguous(x, "x");
    check_cuda_float32_contiguous(y, "y");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto out = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_add_launcher(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), numel, stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_add kernel launch failed");

    return out;
}

// custom_argmin: argmin along last dim, keepdim=True
torch::Tensor custom_argmin(torch::Tensor x) {
    check_cuda_float32_contiguous(x, "x");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");

    int64_t rows = x.size(0);
    int64_t row_len = x.numel() / rows; // since x is contiguous and shape [rows, row_len]

    auto out = torch::empty({rows, 1}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_argmin_launcher(x.data_ptr<float>(), out.data_ptr<int64_t>(), rows, row_len, stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_argmin kernel launch failed");

    return out;
}

// custom_stft: Short-Time Fourier Transform
torch::Tensor custom_stft(torch::Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length, bool return_complex) {
    check_cuda_float32_contiguous(x, "x");
    TORCH_CHECK(return_complex, "return_complex must be true");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, audio_len]");
    TORCH_CHECK(n_fft > 0 && hop_length > 0 && win_length > 0, "n_fft, hop_length, win_length must be positive");

    int64_t batch = x.size(0);
    int64_t audio_len = x.size(1);

    // Compute number of frames as in PyTorch (center=True)
    int64_t pad = n_fft / 2;
    int64_t padded_len = audio_len + 2 * pad;
    int64_t num_frames = 1 + (padded_len - n_fft) / hop_length;
    int64_t n_freqs = n_fft / 2 + 1;

    // Output tensor: complex64 with shape [batch, n_freqs, num_frames]
    auto out = torch::empty({batch, n_freqs, num_frames}, x.options().dtype(torch::kComplexFloat));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    // Call launcher with output as float2 pointer (complex64 is two floats)
    custom_stft_launcher(
        x.data_ptr<float>(),
        reinterpret_cast<float2*>(out.data_ptr<c10::complex<float>>()),
        batch, audio_len, n_fft, hop_length, win_length, stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_stft kernel launch failed");

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_add", &custom_add, "Custom element-wise addition", py::arg("x"), py::arg("y"));
    m.def("custom_argmin", &custom_argmin, "Custom argmin along last dimension", py::arg("x"));
    m.def("custom_stft", &custom_stft, "Custom STFT",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"), py::arg("return_complex"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);