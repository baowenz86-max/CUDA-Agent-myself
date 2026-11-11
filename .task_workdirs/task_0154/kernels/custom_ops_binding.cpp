#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void stft_forward_launcher(
    const float* input,
    float2* output,
    int batch,
    int sequence_length,
    int n_fft,
    int win_length,
    int hop_length,
    cudaStream_t stream
);

extern "C" void view_as_real_permute_launcher(
    const float2* input,
    float* output,
    int batch,
    int freq_bins,
    int frames,
    cudaStream_t stream
);

extern "C" void conv2d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream
);

extern "C" void index_select_forward_launcher(
    const float* input,
    const int64_t* indices,
    float* output,
    int batch,
    int channels,
    int in_dim,
    int frames,
    int num_indices,
    int dim,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// STFT forward
static torch::Tensor stft_forward(torch::Tensor input, int64_t n_fft, int64_t win_length, int64_t hop_length) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D (batch, sequence_length)");

    int batch = input.size(0);
    int sequence_length = input.size(1);
    int freq_bins = n_fft / 2 + 1;
    int pad_left = n_fft / 2;
    int pad_right = n_fft - pad_left;
    int padded_len = sequence_length + pad_left + pad_right;
    int frames = 1 + (padded_len - n_fft) / hop_length; // center=True

    // Allocate output: complex64 [batch, freq_bins, frames]
    auto options = torch::TensorOptions().dtype(torch::kComplexFloat).device(input.device());
    auto output = torch::empty({batch, freq_bins, frames}, options);

    cudaStream_t stream = getCurrentStream();
    stft_forward_launcher(
        input.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr<c10::complex<float>>()),
        batch,
        sequence_length,
        static_cast<int>(n_fft),
        static_cast<int>(win_length),
        static_cast<int>(hop_length),
        stream
    );

    // Check for kernel launch error
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "stft_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// view_as_real_permute forward
static torch::Tensor view_as_real_permute_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kComplexFloat, "input must be complex64");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, freq_bins, frames)");

    int batch = input.size(0);
    int freq_bins = input.size(1);
    int frames = input.size(2);

    // Allocate output: float32 [batch, 2, freq_bins, frames]
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::empty({batch, 2, freq_bins, frames}, options);

    cudaStream_t stream = getCurrentStream();
    view_as_real_permute_launcher(
        reinterpret_cast<const float2*>(input.data_ptr<c10::complex<float>>()),
        output.data_ptr<float>(),
        batch,
        freq_bins,
        frames,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "view_as_real_permute_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// conv2d forward
static torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::tuple<int64_t, int64_t> stride,
    std::tuple<int64_t, int64_t> padding,
    std::tuple<int64_t, int64_t> dilation
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias->size(0) == weight.size(0), "bias size must match out_channels");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (K,C,R,S)");

    // Input may not be contiguous due to permute; cuDNN can handle strides but for simplicity we ensure contiguous
    auto input_contig = input.contiguous();

    int batch = input_contig.size(0);
    int in_channels = input_contig.size(1);
    int in_height = input_contig.size(2);
    int in_width = input_contig.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    int stride_h = static_cast<int>(std::get<0>(stride));
    int stride_w = static_cast<int>(std::get<1>(stride));
    int padding_h = static_cast<int>(std::get<0>(padding));
    int padding_w = static_cast<int>(std::get<1>(padding));
    int dilation_h = static_cast<int>(std::get<0>(dilation));
    int dilation_w = static_cast<int>(std::get<1>(dilation));

    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::empty({batch, out_channels, out_height, out_width}, options);

    cudaStream_t stream = getCurrentStream();
    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;
    conv2d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_height,
        in_width,
        out_channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv2d_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// index_select forward
static torch::Tensor index_select_forward(torch::Tensor input, int64_t dim, torch::Tensor index) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(index.is_cuda(), "index must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(index.dtype() == torch::kInt64, "index must be int64");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (batch, channels, freq_bins, frames)");
    TORCH_CHECK(dim == 2, "Only dim=2 is currently supported");

    int batch = input.size(0);
    int channels = input.size(1);
    int in_dim = input.size(2);
    int frames = input.size(3);
    int num_indices = index.numel();

    // Output shape: [batch, channels, num_indices, frames]
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::empty({batch, channels, num_indices, frames}, options);

    cudaStream_t stream = getCurrentStream();
    index_select_forward_launcher(
        input.data_ptr<float>(),
        index.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_dim,
        frames,
        num_indices,
        static_cast<int>(dim),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "index_select_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("stft_forward", &stft_forward, "STFT forward",
          pybind11::arg("input"), pybind11::arg("n_fft"), pybind11::arg("win_length"), pybind11::arg("hop_length"));
    m.def("view_as_real_permute_forward", &view_as_real_permute_forward, "view_as_real_permute forward",
          pybind11::arg("input"));
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("dilation"));
    m.def("index_select_forward", &index_select_forward, "Index select forward",
          pybind11::arg("input"), pybind11::arg("dim"), pybind11::arg("index"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
