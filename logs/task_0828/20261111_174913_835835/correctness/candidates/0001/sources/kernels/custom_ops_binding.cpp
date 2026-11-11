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
    // num_frames calculation for center=True, pad_mode='reflect'
    int num_frames = (samples + (hop_length - 1)) / hop_length; // since center padding effectively shifts, this matches torch.stft with default settings? Actually torch.stft formula for center=True: num_frames = 1 + (samples + 2*pad)/hop_length, where pad = n_fft//2. For n_fft=4096, pad=2048, samples=16384, hop=512 => (16384+4096)/512 = 40? Wait: (samples + 2*pad) / hop = (16384+4096)/512 = 20480/512 = 40. But reference says 33. Recompute: torch.stft with center=True, for even n_fft, left padding = n_fft//2, right padding = n_fft - n_fft//2 - 1? Actually PyTorch reflect pad for stft uses center=True and pad_mode='reflect', amount = n_fft//2 on both sides? Let's check: samples=16384, n_fft=4096, hop=512. According to PyTorch docs: if center=True, input is padded on both sides so that t-th frame is centered at t*hop. The amount of padding is such that output length = 1 + (input_length + 2*padding - win_length)/hop? No, formula: num_frames = 1 + floor(input_length / hop_length) when center=True and windows of length n_fft are placed at positions t*hop - n_fft/2. Indeed, with center=True, frames start at t*hop - n_fft/2 for t=0,...; the condition for t-th frame to fit: t*hop - n_fft/2 >= -n_fft/2 (always true for t>=0) and t*hop - n_fft/2 + n_fft <= input_length + n_fft/2? Actually after reflection, the padded length is input_length + 2*(n_fft//2) = 16384 + 4096 = 20480. Number of frames = floor((padded_length - n_fft)/hop) + 1 = (20480-4096)/512 +1 = 16384/512+1 = 32+1=33. So num_frames = samples/hop_length + 1 when samples divisible by hop_length. That is 33. Our formula (samples + (hop-1))/hop gives ceil(16384/512)=32, wrong. So special case: center=True, reflect padding, for even n_fft, number of frames = (samples / hop) + 1 if samples % hop == 0? Actually let's derive general: padded_length = samples + n_fft (since pad both sides n_fft//2, total n_fft). number of frames = floor((padded_length - n_fft)/hop) + 1 = floor(samples/hop) + 1. So if samples % hop == 0, that's samples/hop + 1; else floor(samples/hop)+1. That's equivalent to samples/hop + 1 (integer division). So num_frames = samples / hop_length + 1 (for our case 16384/512+1=33). We'll compute as (samples / hop_length) + 1 using integer division.
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

// custom_reshape_output: equivalent to torch.reshape(x, shape), shape is a tuple like (batch, n_freq, -1)
torch::Tensor custom_reshape_output(torch::Tensor x, py::tuple shape) {
    TORCH_CHECK(x.is_cuda(), "custom_reshape_output: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kComplexFloat, "custom_reshape_output: input must be complex64");
    // Convert py::tuple to torch::IntArrayRef
    std::vector<int64_t> shape_vec;
    shape_vec.reserve(shape.size());
    for (auto item : shape) {
        shape_vec.push_back(item.cast<int64_t>());
    }
    return x.reshape(torch::IntArrayRef(shape_vec));
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
