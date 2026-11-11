#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avg_pool1d_launcher(
    const float* x,
    float* y,
    int batch,
    int channels,
    int input_length,
    int output_size,
    cudaStream_t stream);

extern "C" void transpose_1_2_launcher(
    const float* x,
    float* y,
    int batch,
    int dim1,
    int dim2,
    int stride_batch_x,
    int stride_dim1_x,
    int stride_dim2_x,
    cudaStream_t stream);

extern "C" void lstm_forward_launcher_actual(
    const float* x,
    const float* weight_ih_l0,
    const float* weight_hh_l0,
    const float* bias_ih_l0,
    const float* bias_hh_l0,
    const float* weight_ih_l1,
    const float* weight_hh_l1,
    const float* bias_ih_l1,
    const float* bias_hh_l1,
    float* y,
    float* h_n,
    float* c_n,
    int batch,
    int seq_len,
    int input_size,
    int hidden_size,
    cudaStream_t stream);

extern "C" void reshape_2d_launcher_actual(
    const float* x,
    float* y,
    int batch,
    int seq_len,
    int hidden_size,
    int64_t stride_batch_x,
    int64_t stride_seq_x,
    int64_t stride_hidden_x,
    cudaStream_t stream);

extern "C" void linear_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* y,
    int batch,
    int in_features,
    int out_features,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// Binding: adaptive_avg_pool1d
// -----------------------------------------------------------------------------
torch::Tensor adaptive_avg_pool1d(torch::Tensor x, int64_t output_size) {
    TORCH_CHECK(x.is_cuda(), "adaptive_avg_pool1d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "adaptive_avg_pool1d: input must be float32");
    TORCH_CHECK(x.dim() == 3, "adaptive_avg_pool1d: input must be 3D (batch, channels, length)");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t input_length = x.size(2);

    auto y = torch::empty({batch, channels, output_size}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    adaptive_avg_pool1d_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(channels),
        static_cast<int>(input_length),
        static_cast<int>(output_size),
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// Binding: transpose_1_2
// -----------------------------------------------------------------------------
torch::Tensor transpose_1_2(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "transpose_1_2: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "transpose_1_2: input must be float32");
    TORCH_CHECK(x.dim() == 3, "transpose_1_2: input must be 3D (batch, dim1, dim2)");

    int64_t batch = x.size(0);
    int64_t dim1 = x.size(1);
    int64_t dim2 = x.size(2);

    // Transpose dim1 and dim2, output shape [batch, dim2, dim1]
    auto y = torch::empty({batch, dim2, dim1}, x.options());

    // Get strides of input (in elements, not bytes)
    int64_t stride_batch_x = x.stride(0);
    int64_t stride_dim1_x = x.stride(1);
    int64_t stride_dim2_x = x.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    transpose_1_2_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(dim1),
        static_cast<int>(dim2),
        static_cast<int>(stride_batch_x),
        static_cast<int>(stride_dim1_x),
        static_cast<int>(stride_dim2_x),
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// Binding: lstm_forward
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstm_forward(
    torch::Tensor x,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0,
    torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1,
    torch::Tensor bias_hh_l1,
    int64_t hidden_size,
    int64_t num_layers) {

    TORCH_CHECK(x.is_cuda(), "lstm_forward: input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "lstm_forward: input x must be float32");
    TORCH_CHECK(x.dim() == 3, "lstm_forward: input x must be 3D (batch, seq_len, input_size)");
    TORCH_CHECK(num_layers == 2, "lstm_forward: only num_layers=2 is supported");

    int64_t batch = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t input_size = x.size(2);

    // Check weight tensors
    auto check_weight = [](const torch::Tensor& t, const char* name, int64_t rows, int64_t cols) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(t.dim() == 2 && t.size(0) == rows && t.size(1) == cols, name, " shape mismatch");
    };

    int64_t four_hidden = 4 * hidden_size;
    check_weight(weight_ih_l0, "weight_ih_l0", four_hidden, input_size);
    check_weight(weight_hh_l0, "weight_hh_l0", four_hidden, hidden_size);
    check_weight(weight_ih_l1, "weight_ih_l1", four_hidden, hidden_size);
    check_weight(weight_hh_l1, "weight_hh_l1", four_hidden, hidden_size);

    auto check_bias = [](const torch::Tensor& t, const char* name, int64_t size) {
        TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
        TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(t.dim() == 1 && t.size(0) == size, name, " shape mismatch");
    };

    check_bias(bias_ih_l0, "bias_ih_l0", four_hidden);
    check_bias(bias_hh_l0, "bias_hh_l0", four_hidden);
    check_bias(bias_ih_l1, "bias_ih_l1", four_hidden);
    check_bias(bias_hh_l1, "bias_hh_l1", four_hidden);

    // Allocate outputs
    auto y = torch::empty({batch, seq_len, hidden_size}, x.options());
    auto h_n = torch::empty({num_layers, batch, hidden_size}, x.options());
    auto c_n = torch::empty({num_layers, batch, hidden_size}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    lstm_forward_launcher_actual(
        x.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(),
        bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(),
        bias_hh_l1.data_ptr<float>(),
        y.data_ptr<float>(),
        h_n.data_ptr<float>(),
        c_n.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(seq_len),
        static_cast<int>(input_size),
        static_cast<int>(hidden_size),
        stream);

    return std::make_tuple(y, std::make_tuple(h_n, c_n));
}

// -----------------------------------------------------------------------------
// Binding: reshape_2d
// -----------------------------------------------------------------------------
torch::Tensor reshape_2d(torch::Tensor x, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(x.is_cuda(), "reshape_2d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "reshape_2d: input must be float32");
    TORCH_CHECK(x.dim() == 3, "reshape_2d: input must be 3D (batch, seq_len, hidden_size)");

    int64_t batch = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t hidden_size = x.size(2);

    TORCH_CHECK(dim0 == batch, "reshape_2d: dim0 must equal batch size");
    TORCH_CHECK(dim1 == seq_len * hidden_size, "reshape_2d: dim1 must equal seq_len * hidden_size");

    auto y = torch::empty({dim0, dim1}, x.options());

    int64_t stride_batch_x = x.stride(0);
    int64_t stride_seq_x = x.stride(1);
    int64_t stride_hidden_x = x.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reshape_2d_launcher_actual(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(seq_len),
        static_cast<int>(hidden_size),
        stride_batch_x,
        stride_seq_x,
        stride_hidden_x,
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// Binding: linear (bias can be null)
// -----------------------------------------------------------------------------
torch::Tensor linear(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.is_cuda(), "linear: input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "linear: input x must be float32");
    TORCH_CHECK(x.dim() == 2, "linear: input x must be 2D (batch, in_features)");
    TORCH_CHECK(x.is_contiguous(), "linear: input x must be contiguous");

    TORCH_CHECK(weight.is_cuda(), "linear: weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "linear: weight must be float32");
    TORCH_CHECK(weight.dim() == 2, "linear: weight must be 2D (out_features, in_features)");
    TORCH_CHECK(weight.is_contiguous(), "linear: weight must be contiguous");

    int64_t batch = x.size(0);
    int64_t in_features = x.size(1);
    int64_t out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "linear: weight shape mismatch");

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        auto bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "linear: bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "linear: bias must be float32");
        TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == out_features, "linear: bias shape mismatch");
        TORCH_CHECK(bias_tensor.is_contiguous(), "linear: bias must be contiguous");
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    auto y = torch::empty({batch, out_features}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        y.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// Register bindings
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool1d", &adaptive_avg_pool1d, "Adaptive average pooling 1D",
          pybind11::arg("x"), pybind11::arg("output_size"));
    m.def("transpose_1_2", &transpose_1_2, "Transpose dimensions 1 and 2",
          pybind11::arg("x"));
    m.def("lstm_forward", &lstm_forward, "LSTM forward (2 layers)",
          pybind11::arg("x"),
          pybind11::arg("weight_ih_l0"),
          pybind11::arg("weight_hh_l0"),
          pybind11::arg("bias_ih_l0"),
          pybind11::arg("bias_hh_l0"),
          pybind11::arg("weight_ih_l1"),
          pybind11::arg("weight_hh_l1"),
          pybind11::arg("bias_ih_l1"),
          pybind11::arg("bias_hh_l1"),
          pybind11::arg("hidden_size"),
          pybind11::arg("num_layers"));
    m.def("reshape_2d", &reshape_2d, "Reshape 3D to 2D",
          pybind11::arg("x"), pybind11::arg("dim0"), pybind11::arg("dim1"));
    m.def("linear", &linear, "Linear layer",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
