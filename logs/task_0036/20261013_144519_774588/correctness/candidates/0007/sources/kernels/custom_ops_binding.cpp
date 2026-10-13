#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cufft.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void add_launcher(float* out, const float* x, const float* y, int64_t n, cudaStream_t stream);
extern "C" void argmin_launcher(const float* x, int64_t* out, int64_t batch, int64_t length, int64_t x_stride0, cudaStream_t stream);
extern "C" void stft_launcher(const float* x, cufftComplex* out, int64_t batch, int64_t length, int n_fft, int hop_length, int win_length, cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// add: elementwise addition x + y
torch::Tensor add(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");

    auto out = torch::empty_like(x);
    int64_t n = x.numel();

    add_launcher(out.data_ptr<float>(), x.data_ptr<float>(), y.data_ptr<float>(), n, getCurrentStream());

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "add kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// argmin_dim: argmin along last dimension, keepdim=True
torch::Tensor argmin_dim(torch::Tensor x, int64_t dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, length]");
    TORCH_CHECK(dim == -1 || dim == 1, "dim must be -1 or 1 for 2D tensor");
    TORCH_CHECK(keepdim == true, "keepdim must be true");

    int64_t batch = x.size(0);
    int64_t length = x.size(1);
    int64_t x_stride0 = x.stride(0);

    auto out = torch::empty({batch, 1}, x.options().dtype(torch::kInt64));
    // Ensure output is contiguous
    TORCH_CHECK(out.is_contiguous(), "output must be contiguous");

    argmin_launcher(x.data_ptr<float>(), out.data_ptr<int64_t>(), batch, length, x_stride0, getCurrentStream());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "argmin kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// stft_complex: STFT with rectangular window, center=True, pad_mode='reflect', one-sided
torch::Tensor stft_complex(torch::Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length, bool return_complex) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, length]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(return_complex == true, "return_complex must be true");
    TORCH_CHECK(n_fft > 0 && hop_length > 0 && win_length > 0, "n_fft, hop_length, win_length must be positive");
    TORCH_CHECK(win_length <= n_fft, "win_length must be <= n_fft");

    int64_t batch = x.size(0);
    int64_t length = x.size(1);
    int64_t pad = n_fft / 2;
    int64_t n_freq = n_fft / 2 + 1;
    int64_t n_frames = 1 + (length + 2 * pad - n_fft) / hop_length;

    // Output tensor shape [batch, n_freq, n_frames] with same layout as torch.stft
    // PyTorch's stft returns a tensor with strides [n_freq*n_frames, 1, n_freq]
    auto out = torch::empty({batch, n_freq, n_frames}, x.options().dtype(torch::kComplexFloat));
    // Ensure output is contiguous in memory as expected by launcher (batch, n_frames, n_freq)
    // The launcher writes to out as a linear array of complex values with shape [batch, n_frames, n_freq]
    // We need to ensure that out.data_ptr<cufftComplex>() points to contiguous memory in that order.
    // The torch tensor with shape [batch, n_freq, n_frames] is normally contiguous, giving strides [n_freq*n_frames, n_frames, 1]
    // But we need strides [n_freq*n_frames, 1, n_freq]. Actually, we want the memory layout where frame dimension has stride n_freq.
    // That is exactly a tensor with shape [batch, n_frames, n_freq] contiguous, then we can permute to [batch, n_freq, n_frames].
    // However, the launcher expects out pointer to be [batch, n_freq, n_frames] with strides [n_freq*n_frames, 1, n_freq]?
    // The .cu file comment says output stride [401166, 1, 513], meaning batch stride = n_freq*n_frames, freq stride = 1, frame stride = n_freq.
    // This is a non-contiguous tensor. We can create it using torch::empty with strides.
    // We'll create a contiguous tensor with shape [batch, n_frames, n_freq] and then permute it to [batch, n_freq, n_frames].
    // This gives the correct strides. The launcher writes to the contiguous base tensor, then we return the permuted view.
    auto temp = torch::empty({batch, n_frames, n_freq}, x.options().dtype(torch::kComplexFloat));
    // Ensure temp is contiguous
    TORCH_CHECK(temp.is_contiguous(), "temp must be contiguous");
    // Call launcher with temp data pointer
    stft_launcher(x.data_ptr<float>(), reinterpret_cast<cufftComplex*>(temp.data_ptr<c10::complex<float>>()),
                  batch, length, (int)n_fft, (int)hop_length, (int)win_length, getCurrentStream());

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "stft kernel launch failed: ", cudaGetErrorString(err));

    // Permute temp from [batch, n_frames, n_freq] to [batch, n_freq, n_frames]
    auto out = temp.permute({0, 2, 1});
    // Ensure the returned tensor has the exact strides as expected by reference: [n_freq*n_frames, 1, n_freq]
    TORCH_CHECK(out.stride(0) == n_freq * n_frames, "stride0 mismatch");
    TORCH_CHECK(out.stride(1) == 1, "stride1 mismatch");
    TORCH_CHECK(out.stride(2) == n_freq, "stride2 mismatch");

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("add", &add, "Elementwise addition x + y",
          py::arg("x"), py::arg("y"));
    m.def("argmin_dim", &argmin_dim, "Argmin along last dimension with keepdim=True",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("stft_complex", &stft_complex, "STFT with rectangular window, center=True, reflect padding, one-sided",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"), py::arg("return_complex"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
