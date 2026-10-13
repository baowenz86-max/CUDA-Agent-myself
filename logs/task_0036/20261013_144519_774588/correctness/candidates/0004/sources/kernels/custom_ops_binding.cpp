#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <stdexcept>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void custom_add_launcher(const float* x, const float* y, float* out, int64_t n, cudaStream_t stream);
extern "C" void custom_argmin_launcher(const float* x, int64_t* out, int64_t rows, int64_t cols, cudaStream_t stream);
extern "C" void custom_stft_launcher(const float* x, int64_t B, int64_t L, int n_fft, int hop_length, int win_length,
                                    cufftComplex* out, cudaStream_t stream);

// Helper to check CUDA errors
static void check_cuda_error(const char* func_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error after ") + func_name + ": " + cudaGetErrorString(err));
    }
}

// ---------------- custom_add ----------------
torch::Tensor custom_add(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have same shape");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_add_launcher(x.data_ptr<float>(), y.data_ptr<float>(), out.data_ptr<float>(), n, stream);
    check_cuda_error("custom_add_launcher");
    return out;
}

// ---------------- custom_argmin ----------------
torch::Tensor custom_argmin(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");

    int64_t rows = x.dim() == 1 ? 1 : x.size(0);
    int64_t cols = x.dim() == 1 ? x.size(0) : x.size(-1);
    auto out = torch::empty({rows, 1}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_argmin_launcher(x.data_ptr<float>(), out.data_ptr<int64_t>(), rows, cols, stream);
    check_cuda_error("custom_argmin_launcher");

    if (x.dim() == 1) {
        // keepdim=True should return shape [1, 1] to match torch.argmin on 1D with keepdim
        return out;
    } else {
        return out;
    }
}

// ---------------- custom_stft ----------------
torch::Tensor custom_stft(torch::Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length, bool return_complex) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, length]");
    TORCH_CHECK(n_fft > 0, "n_fft must be positive");
    TORCH_CHECK(hop_length > 0, "hop_length must be positive");
    TORCH_CHECK(win_length > 0, "win_length must be positive");
    TORCH_CHECK(win_length <= n_fft, "win_length must be <= n_fft");
    TORCH_CHECK(return_complex, "only return_complex=True is supported");

    int64_t B = x.size(0);
    int64_t L = x.size(1);
    int64_t n_freq = n_fft / 2 + 1;
    // Compute number of frames: center=True, reflect pad n_fft//2 on each side
    int64_t padded_len = L + n_fft;
    int64_t n_frames = 1 + (padded_len - n_fft) / hop_length; // equals 1 + L / hop_length when win_length == n_fft
    // For general win_length, torch.stft uses center=True with reflect pad n_fft//2, win_length window, but here window is rectangular of length win_length.
    // Actually torch.stft with default window=None creates a rectangular window of length win_length if provided, else n_fft.
    // The frame length is win_length, zero-padded to n_fft if win_length < n_fft. In our case win_length=n_fft.
    // So the formula holds.

    // Output complex tensor with shape [B, n_freq, n_frames]
    auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(x.device());
    auto stft_out = torch::empty({B, n_freq, n_frames}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    // Convert complex64 tensor data pointer to cufftComplex* (which is float2)
    cufftComplex* out_ptr = reinterpret_cast<cufftComplex*>(stft_out.data_ptr<c10::complex<float>>());

    // Call launcher with B, L, n_fft, hop_length, win_length, output pointer
    custom_stft_launcher(x.data_ptr<float>(), B, L, (int)n_fft, (int)hop_length, (int)win_length, out_ptr, stream);
    check_cuda_error("custom_stft_launcher");

    return stft_out;
}

// ---------------- Registration ----------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_add", &custom_add, "Add two tensors element-wise",
          py::arg("x"), py::arg("y"));
    m.def("custom_argmin", &custom_argmin, "Argmin along last dim with keepdim",
          py::arg("x"));
    m.def("custom_stft", &custom_stft, "STFT using cuFFT",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"),
          py::arg("return_complex") = true);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
