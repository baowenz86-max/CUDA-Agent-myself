// Binding file for custom_reshape and custom_stft
// This file contains only PyTorch tensor handling and binding code.
// CUDA kernel launchers are declared from custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_stft_launcher(
    const float* input,
    cuComplex* output,
    int64_t batch,
    int64_t num_samples,
    int64_t n_fft,
    int64_t hop_length,
    int64_t win_length,
    cudaStream_t stream
);

// ======================= custom_reshape =======================
// Performs torch.reshape(input, shape) as a view. shape can be a single int (int64_t)
// or a tuple/list of ints (handled via py::tuple overload).

// Overload 1: shape as single int (contract signature)
torch::Tensor custom_reshape_int(torch::Tensor input, int64_t shape) {
    TORCH_CHECK(input.is_cuda(), "custom_reshape: input must be a CUDA tensor");
    // Allow float32 and complex64 as required by contract
    TORCH_CHECK(input.dtype() == torch::kFloat32 || input.dtype() == torch::kComplex64,
                "custom_reshape: input must be float32 or complex64");
    // Single int shape: convert to a one-element array and reshape
    return input.reshape({shape});
}

// Overload 2: shape as tuple (for model_new.py calls like (-1, x.size(-1)))
torch::Tensor custom_reshape_tuple(torch::Tensor input, py::tuple shape) {
    TORCH_CHECK(input.is_cuda(), "custom_reshape: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32 || input.dtype() == torch::kComplex64,
                "custom_reshape: input must be float32 or complex64");
    std::vector<int64_t> dims;
    for (auto item : shape) {
        // Convert each element to int64_t (handles Python ints)
        dims.push_back(py::cast<int64_t>(item));
    }
    return input.reshape(torch::IntArrayRef(dims));
}

// ======================= custom_stft =======================
torch::Tensor custom_stft(
    torch::Tensor input,
    int64_t n_fft,
    int64_t hop_length,
    int64_t win_length,
    bool return_complex
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "custom_stft: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_stft: input must be float32");
    TORCH_CHECK(input.dim() == 2, "custom_stft: input must be 2D [batch, num_samples]");
    TORCH_CHECK(n_fft > 0 && hop_length > 0 && win_length > 0, "custom_stft: n_fft/hop_length/win_length must be positive");
    TORCH_CHECK(win_length <= n_fft, "custom_stft: win_length must be <= n_fft");
    TORCH_CHECK((input.size(1) - n_fft) % hop_length == 0, "custom_stft: (num_samples - n_fft) must be divisible by hop_length");

    // Compute output dimensions
    int64_t batch = input.size(0);
    int64_t num_samples = input.size(1);
    int64_t bins = n_fft / 2 + 1;
    int64_t frames = 1 + (num_samples - n_fft) / hop_length;
    TORCH_CHECK(frames > 0, "custom_stft: too few frames");

    // Allocate output tensor with correct dtype and strides matching torch.stft
    auto options = torch::TensorOptions()
        .dtype(torch::kComplex64)
        .device(input.device());
    torch::Tensor output = torch::empty({batch, bins, frames}, options);

    // Ensure output is contiguous (it should be by default from empty)
    TORCH_CHECK(output.is_contiguous(), "custom_stft: output must be contiguous");

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_stft_launcher(
        input.data_ptr<float>(),
        reinterpret_cast<cuComplex*>(output.data_ptr<c10::complex<float>>()),
        batch,
        num_samples,
        n_fft,
        hop_length,
        win_length,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_stft: CUDA kernel launch failed: ", cudaGetErrorString(err));

    // Synchronize to catch asynchronous errors (optional, may be removed for performance)
    // cudaStreamSynchronize(stream); // Uncomment for debugging, but not needed here.

    return output;
}

// ======================= Registration =======================
void register_custom_ops(pybind11::module& m) {
    // Register custom_reshape with two overloads: int shape and tuple shape.
    // The tuple overload is used by model_new.py as it passes (-1, x.size(-1)).
    m.def("custom_reshape", &custom_reshape_int, "Reshape tensor (single int shape)",
          py::arg("input"), py::arg("shape"));
    m.def("custom_reshape", &custom_reshape_tuple, "Reshape tensor (tuple shape)",
          py::arg("input"), py::arg("shape"));

    // Register custom_stft
    m.def("custom_stft", &custom_stft, "Short-time Fourier transform",
          py::arg("input"), py::arg("n_fft"), py::arg("hop_length"),
          py::arg("win_length"), py::arg("return_complex"));
}

// Auto-register using the fixed infrastructure
REGISTER_BINDING(custom_ops, register_custom_ops);
