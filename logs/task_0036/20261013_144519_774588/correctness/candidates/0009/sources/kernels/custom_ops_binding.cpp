#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cufft.h>
#include <c10/util/complex.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" cudaError_t add_forward_launcher(const float* x,
                                            const float* y,
                                            float* out,
                                            long long n,
                                            cudaStream_t stream);

extern "C" cudaError_t argmin_launcher(const float* x,
                                       long long* out,
                                       int N,
                                       long long B,
                                       cudaStream_t stream);

extern "C" cudaError_t stft_forward_launcher(const float* x,
                                            int B,
                                            int N,
                                            int n_fft,
                                            int hop_length,
                                            int win_length,
                                            cufftComplex* out,
                                            cudaStream_t stream);

// -----------------------------------------------------------------------------
// add_forward: elementwise x + y
// -----------------------------------------------------------------------------
torch::Tensor add_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(y.is_contiguous(), "y must be contiguous");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto output = torch::empty_like(x);
    long long n = x.numel();
    if (n == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = add_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "add_forward_launcher failed with error: ", cudaGetErrorString(err));
    return output;
}

// -----------------------------------------------------------------------------
// argmin_lastdim_keepdim: argmin over last dim, keepdim=True
// -----------------------------------------------------------------------------
torch::Tensor argmin_lastdim_keepdim(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");

    int64_t N = x.size(-1);
    TORCH_CHECK(N > 0, "last dimension size must be positive");
    int64_t B = x.numel() / N;

    auto options = torch::TensorOptions().dtype(torch::kInt64).device(x.device());
    auto output = torch::empty({B, 1}, options);
    if (B == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = argmin_launcher(
        x.data_ptr<float>(),
        reinterpret_cast<long long*>(output.data_ptr<int64_t>()),
        static_cast<int>(N),
        static_cast<long long>(B),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "argmin_launcher failed with error: ", cudaGetErrorString(err));
    return output;
}

// -----------------------------------------------------------------------------
// stft_forward: STFT with default center=True, pad_mode='reflect', onesided=True
// -----------------------------------------------------------------------------
torch::Tensor stft_forward(torch::Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [B, N]");
    TORCH_CHECK(n_fft > 0 && hop_length > 0 && win_length > 0, "n_fft, hop_length, win_length must be positive");
    TORCH_CHECK(win_length <= n_fft, "win_length must be <= n_fft");

    int64_t B = x.size(0);
    int64_t N = x.size(1);
    int64_t F = n_fft / 2 + 1;
    int64_t T = 1 + N / hop_length;

    // Expected stride for torch.stft output: [F*T, 1, F]
    auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(x.device());
    auto output = torch::empty_strided({B, F, T}, {F * T, 1, F}, options);
    if (B == 0 || N == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = stft_forward_launcher(
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(N),
        static_cast<int>(n_fft),
        static_cast<int>(hop_length),
        static_cast<int>(win_length),
        reinterpret_cast<cufftComplex*>(output.data_ptr<c10::complex<float>>()),
        stream
    );
    TORCH_CHECK(err == cudaSuccess, "stft_forward_launcher failed with error: ", cudaGetErrorString(err));
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("add_forward", &add_forward, "Elementwise addition (x + y)",
          py::arg("x"), py::arg("y"));
    m.def("argmin_lastdim_keepdim", &argmin_lastdim_keepdim, "Argmin over last dimension with keepdim",
          py::arg("x"));
    m.def("stft_forward", &stft_forward, "STFT with Hann window, center=True, reflect padding, onesided",
          py::arg("x"), py::arg("n_fft") = 1024, py::arg("hop_length") = 256, py::arg("win_length") = 1024);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
