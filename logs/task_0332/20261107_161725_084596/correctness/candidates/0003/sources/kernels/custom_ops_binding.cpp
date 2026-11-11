#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include <vector>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avg_pool1d_launcher(const float* x_data,
                                              float* y_data,
                                              int batch,
                                              int channels,
                                              int input_length,
                                              int output_length,
                                              int in_stride_batch,
                                              int in_stride_channel,
                                              int in_stride_length,
                                              cudaStream_t stream);

extern "C" void transpose_copy_launcher(const float* x_data,
                                         float* y_data,
                                         int batch,
                                         int dim0_size,
                                         int dim1_size,
                                         int in_stride_batch,
                                         int in_stride_dim0,
                                         int in_stride_dim1,
                                         cudaStream_t stream);

extern "C" void flatten_launcher(const float* x_data,
                                  float* y_data,
                                  int batch,
                                  int seq_len,
                                  int hidden_size,
                                  int in_stride_batch,
                                  int in_stride_seq,
                                  int in_stride_hidden,
                                  cudaStream_t stream);

extern "C" void linear_launcher(const float* x_data,
                                 const float* w_data,
                                 const float* b_data,
                                 float* y_data,
                                 int batch_size,
                                 int in_features,
                                 int out_features,
                                 int has_bias,
                                 cudaStream_t stream);

extern "C" void lstm_launcher(
    const float* x,
    float* y,
    float* h_n,
    float* c_n,
    const float* weight_ih_l0,
    const float* weight_hh_l0,
    const float* bias_ih_l0,
    const float* bias_hh_l0,
    const float* weight_ih_l1,
    const float* weight_hh_l1,
    const float* bias_ih_l1,
    const float* bias_hh_l1,
    int batch_size,
    int seq_len,
    int input_size,
    int hidden_size,
    int num_layers,
    cudaStream_t stream
);

// ------------------------- Helper: get current stream -------------------------
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// ------------------------- AdaptiveAvgPool1d -------------------------
torch::Tensor adaptive_avg_pool1d_forward(torch::Tensor x, int64_t output_size) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "Expected 3D input for this task");
    TORCH_CHECK(output_size > 0, "output_size must be positive");

    int batch = x.size(0);
    int channels = x.size(1);
    int input_length = x.size(2);

    auto y = torch::empty({batch, channels, output_size}, x.options());

    int in_stride_batch = x.stride(0);
    int in_stride_channel = x.stride(1);
    int in_stride_length = x.stride(2);

    cudaStream_t stream = getCurrentStream();
    adaptive_avg_pool1d_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch,
        channels,
        input_length,
        static_cast<int>(output_size),
        in_stride_batch,
        in_stride_channel,
        in_stride_length,
        stream
    );

    return y;
}

// ------------------------- Transpose 2D -------------------------
torch::Tensor transpose_2d_forward(torch::Tensor x, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(dim0 >= 0 && dim1 >= 0 && dim0 < x.dim() && dim1 < x.dim(), "Invalid dims");
    TORCH_CHECK(x.dim() == 3, "Expected 3D input for this task");
    TORCH_CHECK((dim0 == 1 && dim1 == 2) || (dim0 == 2 && dim1 == 1), "Only transpose of dims 1 and 2 supported");

    int batch = x.size(0);
    int dim0_size = x.size(dim0);
    int dim1_size = x.size(dim1);

    // Output shape: swap dim0 and dim1
    std::vector<int64_t> out_shape = {x.size(0), x.size(1), x.size(2)};
    std::swap(out_shape[dim0], out_shape[dim1]);

    auto y = torch::empty(out_shape, x.options());

    int in_stride_batch = x.stride(0);
    int in_stride_dim0 = x.stride(dim0);
    int in_stride_dim1 = x.stride(dim1);

    cudaStream_t stream = getCurrentStream();
    transpose_copy_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch,
        dim0_size,
        dim1_size,
        in_stride_batch,
        in_stride_dim0,
        in_stride_dim1,
        stream
    );

    return y;
}

// ------------------------- Flatten -------------------------
torch::Tensor flatten_forward(torch::Tensor x, int64_t start_dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(start_dim >= 0 && start_dim < x.dim(), "Invalid start_dim");
    TORCH_CHECK(x.dim() == 3, "Expected 3D input for this task");
    TORCH_CHECK(start_dim == 1, "Only start_dim=1 supported for this task");

    int batch = x.size(0);
    int seq_len = x.size(1);
    int hidden_size = x.size(2);

    auto y = torch::empty({batch, seq_len * hidden_size}, x.options());

    int in_stride_batch = x.stride(0);
    int in_stride_seq = x.stride(1);
    int in_stride_hidden = x.stride(2);

    cudaStream_t stream = getCurrentStream();
    flatten_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        batch,
        seq_len,
        hidden_size,
        in_stride_batch,
        in_stride_seq,
        in_stride_hidden,
        stream
    );

    return y;
}

// ------------------------- Linear -------------------------
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32, "Must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D [batch, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int out_features = weight.size(0);
    TORCH_CHECK(in_features == weight.size(1), "Inner dimensions mismatch");
    bool has_bias = bias.defined();
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32 && bias.dim() == 1 && bias.size(0) == out_features,
                    "bias must be 1D float32 CUDA tensor of size out_features");
    }

    auto y = torch::empty({batch_size, out_features}, x.options());

    cudaStream_t stream = getCurrentStream();
    linear_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        has_bias ? bias.data_ptr<float>() : nullptr,
        y.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        has_bias ? 1 : 0,
        stream
    );

    return y;
}

// ------------------------- LSTM -------------------------
// Binding for lstm_forward: accept module as py::object and extract parameters
torch::Tensor lstm_forward(torch::Tensor x, py::object module) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32, "x must be CUDA float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, seq_len, input_size]");

    int batch_size = x.size(0);
    int seq_len = x.size(1);
    int input_size = x.size(2);
    int hidden_size = 256;  // from task contract
    int num_layers = 2;

    // Extract parameters
    auto weight_ih_l0 = module.attr("weight_ih_l0").cast<torch::Tensor>().cuda();
    auto weight_hh_l0 = module.attr("weight_hh_l0").cast<torch::Tensor>().cuda();
    auto bias_ih_l0 = module.attr("bias_ih_l0").cast<torch::Tensor>().cuda();
    auto bias_hh_l0 = module.attr("bias_hh_l0").cast<torch::Tensor>().cuda();
    auto weight_ih_l1 = module.attr("weight_ih_l1").cast<torch::Tensor>().cuda();
    auto weight_hh_l1 = module.attr("weight_hh_l1").cast<torch::Tensor>().cuda();
    auto bias_ih_l1 = module.attr("bias_ih_l1").cast<torch::Tensor>().cuda();
    auto bias_hh_l1 = module.attr("bias_hh_l1").cast<torch::Tensor>().cuda();

    // Allocate output tensors
    auto y = torch::empty({batch_size, seq_len, hidden_size}, x.options());
    auto h_n = torch::empty({num_layers, batch_size, hidden_size}, x.options());
    auto c_n = torch::empty({num_layers, batch_size, hidden_size}, x.options());

    cudaStream_t stream = getCurrentStream();
    lstm_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        h_n.data_ptr<float>(),
        c_n.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(),
        bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(),
        bias_hh_l1.data_ptr<float>(),
        batch_size,
        seq_len,
        input_size,
        hidden_size,
        num_layers,
        stream
    );

    // Return tuple (y, (h_n, c_n))
    return py::make_tuple(y, py::make_tuple(h_n, c_n));
}

// ------------------------- Registration -------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool1d_forward", &adaptive_avg_pool1d_forward, "AdaptiveAvgPool1d forward",
          py::arg("x"), py::arg("output_size"));
    m.def("transpose_2d_forward", &transpose_2d_forward, "Transpose 2D forward",
          py::arg("x"), py::arg("dim0"), py::arg("dim1"));
    m.def("lstm_forward", &lstm_forward, "LSTM forward",
          py::arg("x"), py::arg("module"));
    m.def("flatten_forward", &flatten_forward, "Flatten forward",
          py::arg("x"), py::arg("start_dim"));
    m.def("linear_forward", &linear_forward, "Linear forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
