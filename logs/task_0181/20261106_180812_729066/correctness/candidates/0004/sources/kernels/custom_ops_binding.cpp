#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void argsort_forward_launcher(
    const float* input,
    int64_t* output,
    int batch,
    int channels,
    int seq_len,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    cudaStream_t stream);

extern "C" void constant_pad1d_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_seq_len,
    int out_seq_len,
    int pad_left,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    float value,
    cudaStream_t stream);

extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int seq_len,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    cudaStream_t stream);

// Helper: get current CUDA stream
static cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// ---------------------------------------------------------------------
// argsort_forward
// ---------------------------------------------------------------------
torch::Tensor argsort_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "argsort_forward: input must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "argsort_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "argsort_forward: input must have 3 dims");

    int batch = x.size(0);
    int channels = x.size(1);
    int seq_len = x.size(2);

    auto output = torch::empty_like(x, torch::TensorOptions().dtype(torch::kInt64));

    // Input and output strides (output is contiguous by default, but we pass actual strides)
    int64_t in_stride_batch = x.stride(0);
    int64_t in_stride_channel = x.stride(1);
    int64_t in_stride_seq = x.stride(2);

    int64_t out_stride_batch = output.stride(0);
    int64_t out_stride_channel = output.stride(1);
    int64_t out_stride_seq = output.stride(2);

    cudaStream_t stream = get_current_stream();
    argsort_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        batch, channels, seq_len,
        in_stride_batch, in_stride_channel, in_stride_seq,
        out_stride_batch, out_stride_channel, out_stride_seq,
        stream);

    return output;
}

// ---------------------------------------------------------------------
// constant_pad1d_forward
// ---------------------------------------------------------------------
torch::Tensor constant_pad1d_forward(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    double value) {
    TORCH_CHECK(x.is_cuda(), "constant_pad1d_forward: input must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "constant_pad1d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "constant_pad1d_forward: input must have 3 dims");

    int batch = x.size(0);
    int channels = x.size(1);
    int in_seq_len = x.size(2);
    int out_seq_len = in_seq_len + static_cast<int>(pad_left) + static_cast<int>(pad_right);

    TORCH_CHECK(pad_left >= 0 && pad_right >= 0, "pad must be non-negative");

    auto output = torch::empty(
        {batch, channels, out_seq_len},
        x.options());

    int64_t in_stride_batch = x.stride(0);
    int64_t in_stride_channel = x.stride(1);
    int64_t in_stride_seq = x.stride(2);

    int64_t out_stride_batch = output.stride(0);
    int64_t out_stride_channel = output.stride(1);
    int64_t out_stride_seq = output.stride(2);

    cudaStream_t stream = get_current_stream();
    constant_pad1d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, in_seq_len, out_seq_len,
        static_cast<int>(pad_left),
        in_stride_batch, in_stride_channel, in_stride_seq,
        out_stride_batch, out_stride_channel, out_stride_seq,
        static_cast<float>(value),
        stream);

    return output;
}

// ---------------------------------------------------------------------
// log_softmax_forward
// ---------------------------------------------------------------------
torch::Tensor log_softmax_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "log_softmax_forward: input must be CUDA");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "log_softmax_forward: input must be float32");
    TORCH_CHECK(x.dim() == 3, "log_softmax_forward: input must have 3 dims");

    int batch = x.size(0);
    int channels = x.size(1);
    int seq_len = x.size(2);

    auto output = torch::empty_like(x);

    int64_t in_stride_batch = x.stride(0);
    int64_t in_stride_channel = x.stride(1);
    int64_t in_stride_seq = x.stride(2);

    int64_t out_stride_batch = output.stride(0);
    int64_t out_stride_channel = output.stride(1);
    int64_t out_stride_seq = output.stride(2);

    cudaStream_t stream = get_current_stream();
    log_softmax_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, seq_len,
        in_stride_batch, in_stride_channel, in_stride_seq,
        out_stride_batch, out_stride_channel, out_stride_seq,
        stream);

    return output;
}

// ---------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("argsort_forward", &argsort_forward, "Stable argsort along last dimension");
    m.def("constant_pad1d_forward", &constant_pad1d_forward,
          "Constant pad along last dimension",
          py::arg("x"), py::arg("pad_left"), py::arg("pad_right"), py::arg("value"));
    m.def("log_softmax_forward", &log_softmax_forward, "Log softmax along last dimension");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
