#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" cudaError_t layer_norm_default_launcher(
    const float* x, float* y, int N, int rows, int64_t row_stride,
    int64_t out_row_stride, float eps, cudaStream_t stream);

extern "C" cudaError_t layer_norm_affine_launcher(
    const float* x, float* y, const float* weight, const float* bias,
    int N, int rows, int64_t row_stride, int64_t out_row_stride,
    float eps, cudaStream_t stream);

extern "C" cudaError_t nan_to_num_launcher(
    float* y, const float* x, int64_t n, cudaStream_t stream);

extern "C" cudaError_t add_bias_launcher(
    float* y, const float* x, float bias, int64_t n, cudaStream_t stream);

extern "C" cudaError_t clamp_launcher(
    float* y, const float* x, float min_val, float max_val, int64_t n, cudaStream_t stream);

extern "C" cudaError_t lstm_forward(
    const float* x, float* output, float* h_n, float* c_n,
    const float* w_ih_l0, const float* w_hh_l0,
    const float* b_ih_l0, const float* b_hh_l0,
    const float* w_ih_l1, const float* w_hh_l1,
    const float* b_ih_l1, const float* b_hh_l1,
    int seq_len, int batch_size, int input_size, int hidden_size,
    int num_layers, cudaStream_t stream);

// Helper to check CUDA tensor, float32, and ensure contiguous
torch::Tensor check_cuda_float32_contiguous(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

// layer_norm_default: x (..., N), normalized_shape is last dim size, eps
torch::Tensor layer_norm_default(torch::Tensor x, int64_t normalized_shape, double eps) {
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    TORCH_CHECK(normalized_shape > 0, "normalized_shape must be positive");
    TORCH_CHECK(x.size(-1) == normalized_shape, "normalized_shape must match last dim");
    x = check_cuda_float32_contiguous(x, "x");

    auto y = torch::empty_like(x);
    int N = static_cast<int>(normalized_shape);
    int64_t numel = x.numel();
    int rows = static_cast<int>(numel / N);
    int64_t row_stride = N; // after contiguous, stride last dim = 1, row stride = N
    int64_t out_row_stride = N;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = layer_norm_default_launcher(
        x.data_ptr<float>(), y.data_ptr<float>(), N, rows,
        row_stride, out_row_stride, static_cast<float>(eps), stream);
    TORCH_CHECK(err == cudaSuccess, "layer_norm_default_launcher failed: ", cudaGetErrorString(err));
    return y;
}

// layer_norm_affine: x (..., N), weight (N), bias (N)
torch::Tensor layer_norm_affine(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dimension");
    auto W = check_cuda_float32_contiguous(weight, "weight");
    auto B = check_cuda_float32_contiguous(bias, "bias");
    TORCH_CHECK(W.dim() == 1 && W.size(0) == x.size(-1), "weight shape mismatch");
    TORCH_CHECK(B.dim() == 1 && B.size(0) == x.size(-1), "bias shape mismatch");
    x = check_cuda_float32_contiguous(x, "x");

    auto y = torch::empty_like(x);
    int N = static_cast<int>(x.size(-1));
    int64_t numel = x.numel();
    int rows = static_cast<int>(numel / N);
    int64_t row_stride = N;
    int64_t out_row_stride = N;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = layer_norm_affine_launcher(
        x.data_ptr<float>(), y.data_ptr<float>(),
        W.data_ptr<float>(), B.data_ptr<float>(),
        N, rows, row_stride, out_row_stride, static_cast<float>(eps), stream);
    TORCH_CHECK(err == cudaSuccess, "layer_norm_affine_launcher failed: ", cudaGetErrorString(err));
    return y;
}

// nan_to_num
torch::Tensor nan_to_num(torch::Tensor x) {
    x = check_cuda_float32_contiguous(x, "x");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = nan_to_num_launcher(
        y.data_ptr<float>(), x.data_ptr<float>(), n, stream);
    TORCH_CHECK(err == cudaSuccess, "nan_to_num_launcher failed: ", cudaGetErrorString(err));
    return y;
}

// add_bias: x (...,), bias scalar tensor
torch::Tensor add_bias(torch::Tensor x, torch::Tensor bias) {
    x = check_cuda_float32_contiguous(x, "x");
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32 && bias.numel() == 1,
                "bias must be a CUDA float32 scalar tensor");
    bias = bias.contiguous();
    float bias_val = bias.data_ptr<float>()[0];
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = add_bias_launcher(
        y.data_ptr<float>(), x.data_ptr<float>(), bias_val, n, stream);
    TORCH_CHECK(err == cudaSuccess, "add_bias_launcher failed: ", cudaGetErrorString(err));
    return y;
}

// clamp
torch::Tensor clamp(torch::Tensor x, double min_val, double max_val) {
    x = check_cuda_float32_contiguous(x, "x");
    auto y = torch::empty_like(x);
    int64_t n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = clamp_launcher(
        y.data_ptr<float>(), x.data_ptr<float>(),
        static_cast<float>(min_val), static_cast<float>(max_val), n, stream);
    TORCH_CHECK(err == cudaSuccess, "clamp_launcher failed: ", cudaGetErrorString(err));
    return y;
}

// lstm: x (seq_len, batch, input_size), weights per layer, num_layers
std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstm(
    torch::Tensor x,
    torch::Tensor weight_ih_l0, torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0, torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1, torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1, torch::Tensor bias_hh_l1,
    int64_t num_layers) {
    TORCH_CHECK(x.dim() == 3, "x must be 3D (seq_len, batch, input_size)");
    TORCH_CHECK(num_layers >= 1 && num_layers <= 2, "only support 1 or 2 layers in this contract");
    x = check_cuda_float32_contiguous(x, "x");

    auto w_ih_l0 = check_cuda_float32_contiguous(weight_ih_l0, "weight_ih_l0");
    auto w_hh_l0 = check_cuda_float32_contiguous(weight_hh_l0, "weight_hh_l0");
    auto b_ih_l0 = check_cuda_float32_contiguous(bias_ih_l0, "bias_ih_l0");
    auto b_hh_l0 = check_cuda_float32_contiguous(bias_hh_l0, "bias_hh_l0");

    torch::Tensor w_ih_l1, w_hh_l1, b_ih_l1, b_hh_l1;
    if (num_layers > 1) {
        w_ih_l1 = check_cuda_float32_contiguous(weight_ih_l1, "weight_ih_l1");
        w_hh_l1 = check_cuda_float32_contiguous(weight_hh_l1, "weight_hh_l1");
        b_ih_l1 = check_cuda_float32_contiguous(bias_ih_l1, "bias_ih_l1");
        b_hh_l1 = check_cuda_float32_contiguous(bias_hh_l1, "bias_hh_l1");
    } else {
        // Provide dummy pointers if launcher expects non-null, but we pass nullptr in launcher call.
        // In launcher, we must handle nullptr for layer1 if num_layers==1.
        // See custom_ops.cu: we check num_layers > 1 before using layer1 pointers.
        // So we can pass nullptr.
        w_ih_l1 = torch::Tensor();
        w_hh_l1 = torch::Tensor();
        b_ih_l1 = torch::Tensor();
        b_hh_l1 = torch::Tensor();
    }

    int64_t seq_len = x.size(0);
    int64_t batch_size = x.size(1);
    int64_t input_size = x.size(2);
    int64_t hidden_size = w_hh_l0.size(1); // assuming w_hh_l0 shape [4*hidden, hidden]
    TORCH_CHECK(w_hh_l0.size(0) == 4 * hidden_size, "weight_hh_l0 shape mismatch");
    TORCH_CHECK(w_ih_l0.size(0) == 4 * hidden_size && w_ih_l0.size(1) == input_size, "weight_ih_l0 shape mismatch");
    TORCH_CHECK(b_ih_l0.size(0) == 4 * hidden_size, "bias_ih_l0 shape mismatch");
    TORCH_CHECK(b_hh_l0.size(0) == 4 * hidden_size, "bias_hh_l0 shape mismatch");

    auto output = torch::empty({seq_len, batch_size, hidden_size}, x.options());
    auto h_n = torch::empty({num_layers, batch_size, hidden_size}, x.options());
    auto c_n = torch::empty({num_layers, batch_size, hidden_size}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaError_t err = lstm_forward(
        x.data_ptr<float>(), output.data_ptr<float>(),
        h_n.data_ptr<float>(), c_n.data_ptr<float>(),
        w_ih_l0.data_ptr<float>(), w_hh_l0.data_ptr<float>(),
        b_ih_l0.data_ptr<float>(), b_hh_l0.data_ptr<float>(),
        (num_layers > 1) ? w_ih_l1.data_ptr<float>() : nullptr,
        (num_layers > 1) ? w_hh_l1.data_ptr<float>() : nullptr,
        (num_layers > 1) ? b_ih_l1.data_ptr<float>() : nullptr,
        (num_layers > 1) ? b_hh_l1.data_ptr<float>() : nullptr,
        static_cast<int>(seq_len), static_cast<int>(batch_size),
        static_cast<int>(input_size), static_cast<int>(hidden_size),
        static_cast<int>(num_layers), stream);
    TORCH_CHECK(err == cudaSuccess, "lstm_forward failed: ", cudaGetErrorString(err));

    return std::make_tuple(output, std::make_tuple(h_n, c_n));
}

void register_custom_ops(pybind11::module& m) {
    m.def("layer_norm_default", &layer_norm_default, "LayerNorm without affine",
          pybind11::arg("x"), pybind11::arg("normalized_shape"), pybind11::arg("eps"));
    m.def("layer_norm_affine", &layer_norm_affine, "LayerNorm with affine",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"), pybind11::arg("eps"));
    m.def("nan_to_num", &nan_to_num, "Replace NaN with 0", pybind11::arg("x"));
    m.def("lstm", &lstm, "LSTM forward",
          pybind11::arg("x"),
          pybind11::arg("weight_ih_l0"), pybind11::arg("weight_hh_l0"),
          pybind11::arg("bias_ih_l0"), pybind11::arg("bias_hh_l0"),
          pybind11::arg("weight_ih_l1"), pybind11::arg("weight_hh_l1"),
          pybind11::arg("bias_ih_l1"), pybind11::arg("bias_hh_l1"),
          pybind11::arg("num_layers"));
    m.def("add_bias", &add_bias, "Add scalar bias",
          pybind11::arg("x"), pybind11::arg("bias"));
    m.def("clamp", &clamp, "Clamp tensor",
          pybind11::arg("x"), pybind11::arg("min"), pybind11::arg("max"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
