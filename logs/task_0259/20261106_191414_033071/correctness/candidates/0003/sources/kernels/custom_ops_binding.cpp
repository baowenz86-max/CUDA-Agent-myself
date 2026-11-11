#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void stft_forward_launcher(
    const float* input,
    float2* output,
    int batch,
    int signal_length,
    int n_fft,
    int hop_length,
    cudaStream_t stream
);

extern "C" void reshape_forward_launcher(
    const float2* input,
    float2* output,
    int total,
    cudaStream_t stream
);

static cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

torch::Tensor stft_forward(torch::Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length) {
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D (batch, signal_length)");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input dtype must be float32");

    if (!x.is_cuda()) {
        x = x.to(torch::kCUDA);
    }
    TORCH_CHECK(x.is_cuda(), "Input must be on CUDA after transfer");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int batch = static_cast<int>(x.size(0));
    int signal_length = static_cast<int>(x.size(1));

    int n_freq = static_cast<int>(n_fft / 2 + 1);
    int n_frames = 1 + signal_length / static_cast<int>(hop_length);

    // Allocate contiguous storage
    auto storage = torch::empty({batch, n_freq, n_frames},
                                torch::TensorOptions().dtype(torch::kComplexFloat).device(x.device()));

    // Create a view with expected stride [n_freq*n_frames, 1, n_freq]
    std::vector<int64_t> output_shape = {batch, n_freq, n_frames};
    std::vector<int64_t> output_strides = {static_cast<int64_t>(n_freq) * n_frames, 1, static_cast<int64_t>(n_freq)};
    auto output = storage.as_strided(output_shape, output_strides);

    const float* input_ptr = x.data_ptr<float>();
    float2* output_ptr = reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>());

    cudaStream_t stream = get_current_stream();

    stft_forward_launcher(input_ptr, output_ptr, batch, signal_length,
                          static_cast<int>(n_fft), static_cast<int>(hop_length), stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in stft_forward_launcher: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor reshape_forward(torch::Tensor x, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kComplexFloat, "Input dtype must be complex64");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, n_freq, n_frames)");

    int64_t total = x.numel();
    int64_t dim2 = total / (dim0 * dim1);
    TORCH_CHECK(dim0 * dim1 * dim2 == total, "Given dim0 and dim1 do not evenly divide input elements");

    std::vector<int64_t> output_shape = {dim0, dim1, dim2};
    // Expected stride: [dim1*dim2, 1, dim1]
    std::vector<int64_t> output_strides = {dim1 * dim2, 1, dim1};
    auto output = x.as_strided(output_shape, output_strides);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("stft_forward", &stft_forward,
          "Compute STFT forward",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"));
    m.def("reshape_forward", &reshape_forward,
          "Reshape tensor to [dim0, dim1, -1]",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
