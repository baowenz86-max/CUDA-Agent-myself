#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_stft_launcher(
    const float* input,
    float2* output,
    int batch,
    int L,
    int n_fft,
    int hop_length,
    int win_length,
    cudaStream_t stream
);

// custom_reshape: mirrors torch.reshape, returns a view when possible
torch::Tensor custom_reshape(torch::Tensor input, std::vector<int64_t> shape) {
    TORCH_CHECK(input.is_cuda(), "custom_reshape: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_reshape: input must be float32");
    return input.view(shape);
}

// custom_stft: mirrors torch.stft with return_complex=True and default window=None
// (Hann window generated inside launcher)
torch::Tensor custom_stft(
    torch::Tensor input,
    int64_t n_fft,
    int64_t hop_length,
    int64_t win_length,
    bool return_complex
) {
    TORCH_CHECK(input.is_cuda(), "custom_stft: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_stft: input must be float32");
    TORCH_CHECK(input.dim() == 2, "custom_stft: input must be 2D (batch, L)");
    TORCH_CHECK(n_fft > 0, "custom_stft: n_fft must be positive");
    TORCH_CHECK(hop_length > 0, "custom_stft: hop_length must be positive");
    TORCH_CHECK(win_length == n_fft, "custom_stft: win_length must equal n_fft in this implementation");
    TORCH_CHECK(return_complex, "custom_stft: return_complex must be true");

    // Input is expected to be contiguous; if not, make it contiguous to simplify launcher
    auto input_contig = input.contiguous();

    const int64_t batch = input_contig.size(0);
    const int64_t L = input_contig.size(1);
    // Formula from PyTorch: num_frames = 1 + (L - n_fft) / hop_length (center=True, pad_mode='reflect')
    const int64_t num_frames = 1 + (L - n_fft) / hop_length;
    TORCH_CHECK(num_frames > 0, "custom_stft: invalid parameters, num_frames must be positive");

    const int64_t n_freq = n_fft / 2 + 1;
    // Output tensor: (batch, n_freq, num_frames), dtype complex64
    auto output = torch::empty({batch, n_freq, num_frames}, input_contig.options().dtype(torch::kComplex64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_stft_launcher(
        input_contig.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>()),
        static_cast<int>(batch),
        static_cast<int>(L),
        static_cast<int>(n_fft),
        static_cast<int>(hop_length),
        static_cast<int>(win_length),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_stft: CUDA error after launch: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_reshape", &custom_reshape,
          "Reshape operation (view)",
          pybind11::arg("input"), pybind11::arg("shape"));
    m.def("custom_stft", &custom_stft,
          "Short-time Fourier transform (complex output)",
          pybind11::arg("input"),
          pybind11::arg("n_fft"),
          pybind11::arg("hop_length"),
          pybind11::arg("win_length"),
          pybind11::arg("return_complex"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
