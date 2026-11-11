// Custom ops binding for model_new.py
// Includes wrappers for move_to_cuda, stft_abs, and reflection_pad_2d

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void move_to_cuda_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream);
extern "C" void stft_abs_launcher(float* output, const float* input, int64_t batch_size, int64_t length, int n_fft, int hop_length, int win_length, cudaStream_t stream);
extern "C" void reflection_pad_2d_launcher(float* output, const float* input, int64_t batch_size, int64_t height, int64_t width, int pad_left, int pad_right, int pad_top, int pad_bottom, const int64_t* input_strides, cudaStream_t stream);

// move_to_cuda: copies CPU tensor to GPU
torch::Tensor move_to_cuda(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cpu(), "move_to_cuda: input must be CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "move_to_cuda: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "move_to_cuda: input must be contiguous");

    auto output = torch::empty_like(x, x.options().device(torch::kCUDA));
    auto numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    move_to_cuda_launcher(output.data_ptr<float>(), x.data_ptr<float>(), numel, stream);

    return output;
}

// stft_abs: computes STFT magnitude
torch::Tensor stft_abs(torch::Tensor x, int64_t n_fft, int64_t hop_length, int64_t win_length) {
    TORCH_CHECK(x.is_cuda(), "stft_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "stft_abs: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "stft_abs: input must be contiguous");
    TORCH_CHECK(x.dim() == 2, "stft_abs: input must be 2D [batch, length]");

    int64_t batch_size = x.size(0);
    int64_t length = x.size(1);

    // Compute output shape
    int64_t center_pad = n_fft / 2;
    int64_t num_frames = 1 + (length + 2 * center_pad - n_fft) / hop_length;
    int64_t num_freq = n_fft / 2 + 1;

    auto output = torch::empty({batch_size, num_freq, num_frames}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    stft_abs_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch_size,
        length,
        (int)n_fft,
        (int)hop_length,
        (int)win_length,
        stream
    );

    return output;
}

// reflection_pad_2d: applies reflection padding on last two dims
torch::Tensor reflection_pad_2d(torch::Tensor x, torch::Tensor padding) {
    TORCH_CHECK(x.is_cuda(), "reflection_pad_2d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "reflection_pad_2d: input must be float32");
    TORCH_CHECK(x.dim() >= 3, "reflection_pad_2d: input must have at least 3 dims");
    TORCH_CHECK(padding.is_cpu(), "reflection_pad_2d: padding must be CPU tensor");
    TORCH_CHECK(padding.dtype() == torch::kInt64, "reflection_pad_2d: padding must be int64");
    TORCH_CHECK(padding.numel() == 4, "reflection_pad_2d: padding must have 4 elements");

    // Extract padding values
    auto padding_acc = padding.accessor<int64_t, 1>();
    int64_t pad_left = padding_acc[0];
    int64_t pad_right = padding_acc[1];
    int64_t pad_top = padding_acc[2];
    int64_t pad_bottom = padding_acc[3];

    int64_t batch_size = x.size(0);
    int64_t height = x.size(2);
    int64_t width = x.size(3);

    int64_t out_height = height + pad_top + pad_bottom;
    int64_t out_width = width + pad_left + pad_right;

    auto output = torch::empty({batch_size, x.size(1), out_height, out_width}, x.options());

    // Input strides in elements (not bytes)
    std::vector<int64_t> input_strides_vec = {
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3)
    };

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad_2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        batch_size,
        height,
        width,
        (int)pad_left,
        (int)pad_right,
        (int)pad_top,
        (int)pad_bottom,
        input_strides_vec.data(),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("move_to_cuda", &move_to_cuda, "Copy CPU tensor to CUDA",
          py::arg("x"));
    m.def("stft_abs", &stft_abs, "Compute STFT magnitude",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"), py::arg("win_length"));
    m.def("reflection_pad_2d", &reflection_pad_2d, "Reflection pad last two dims",
          py::arg("x"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
