// Binding for custom STFT and reshape operations. No CUDA kernels here.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_stft_launcher(
    const float* input_ptr,
    float2* output_ptr,
    int batch,
    int samples,
    int n_fft,
    int hop_length,
    int num_frames,
    cudaStream_t stream
);

// custom_reshape_input: equivalent to torch.reshape(x, (-1, x.size(-1)))
torch::Tensor custom_reshape_input(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_reshape_input: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_reshape_input: input must be float32");
    int64_t last_dim = x.size(-1);
    return x.reshape({-1, last_dim});
}

// custom_stft: compute STFT using CUDA launcher.
torch::Tensor custom_stft(torch::Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length) {
    TORCH_CHECK(x.is_cuda(), "custom_stft: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_stft: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "custom_stft: input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "custom_stft: input must be 2D [batch, samples]");
    TORCH_CHECK(win_length == n_fft, "custom_stft: win_length must equal n_fft in current implementation");
    
    int batch = x.size(0);
    int samples = x.size(1);
    int n_freq = n_fft / 2 + 1;
    // num_frames calculation matching torch.stft with center=True, pad_mode='reflect', even n_fft:
    // padded length = samples + n_fft, number of frames = (padded_length - n_fft) / hop_length + 1 = samples / hop_length + 1
    int num_frames_calc = samples / hop_length + 1;
    
    // Allocate output as complex64 tensor of shape [batch, n_freq, num_frames]
    auto output = torch::empty({batch, n_freq, num_frames_calc},
                               torch::TensorOptions().dtype(torch::kComplexFloat).device(x.device()));
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher with raw data pointers
    custom_stft_launcher(
        x.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>()),
        batch,
        samples,
        static_cast<int>(n_fft),
        static_cast<int>(hop_length),
        num_frames_calc,
        stream
    );
    
    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_stft: CUDA kernel launch failed: ", cudaGetErrorString(err));
    
    return output;
}

// custom_reshape_output: reshape to (batch, n_freq, -1); shape argument is unused placeholder to match contract.
torch::Tensor custom_reshape_output(torch::Tensor x, c10::optional<int64_t> shape) {
    TORCH_CHECK(x.is_cuda(), "custom_reshape_output: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kComplexFloat, "custom_reshape_output: input must be complex64");
    TORCH_CHECK(x.dim() == 3, "custom_reshape_output: input must be 3D [batch, n_freq, n_frames]");
    
    int64_t batch = x.size(0);
    int64_t n_freq = x.size(1);
    return x.reshape({batch, n_freq, -1});
}

// Register all custom ops
void register_custom_ops(pybind11::module& m) {
    m.def("custom_reshape_input", &custom_reshape_input, "Reshape input (flatten leading dims)",
          py::arg("x"));
    m.def("custom_stft", &custom_stft, "Compute STFT (complex output)",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"));
    m.def("custom_reshape_output", &custom_reshape_output, "Reshape output (view with inferred dimension)",
          py::arg("x"), py::arg("shape"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
