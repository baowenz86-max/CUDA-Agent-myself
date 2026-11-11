#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void zeros_like_alloc_launcher(
    float* output,
    int64_t num_layers,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream
);

extern "C" void linear_fc_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream
);

extern "C" void lstm_forward_launcher(
    const float* x,
    const float* h0,
    const float* c0,
    const float* w_ih_l0,
    const float* w_hh_l0,
    const float* b_ih_l0,
    const float* b_hh_l0,
    const float* w_ih_l1,
    const float* w_hh_l1,
    const float* b_ih_l1,
    const float* b_hh_l1,
    float* output,
    float* h_n,
    float* c_n,
    int seq_len,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    cudaStream_t stream
);

// Helper: ensure tensor is contiguous float32 CUDA
torch::Tensor ensure_contiguous_float32_cuda(torch::Tensor tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must have dtype torch.float32");
    if (!tensor.is_contiguous()) {
        tensor = tensor.contiguous();
    }
    return tensor;
}

// Operator 1: zeros_like_alloc
torch::Tensor zeros_like_alloc_forward(int64_t num_layers, int64_t batch_size, int64_t hidden_size, int64_t device) {
    TORCH_CHECK(num_layers > 0, "num_layers must be positive");
    TORCH_CHECK(batch_size > 0, "batch_size must be positive");
    TORCH_CHECK(hidden_size > 0, "hidden_size must be positive");
    TORCH_CHECK(device >= 0, "device index must be non-negative");

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::Device(torch::kCUDA, device));

    auto output = torch::empty({num_layers, batch_size, hidden_size}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    zeros_like_alloc_launcher(
        output.data_ptr<float>(),
        num_layers,
        batch_size,
        hidden_size,
        stream
    );

    return output;
}

// Operator 2: linear_fc
torch::Tensor linear_fc_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    input = ensure_contiguous_float32_cuda(input, "input");
    weight = ensure_contiguous_float32_cuda(weight, "weight");
    bias = ensure_contiguous_float32_cuda(bias, "bias");

    // Check dimensions
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, hidden_size]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, hidden_size]");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D [out_features]");

    int64_t batch_size = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == in_features, "weight second dimension must match input hidden size");
    TORCH_CHECK(bias.size(0) == out_features, "bias size must match out_features");

    auto output = torch::empty({batch_size, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_fc_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)batch_size,
        (int)in_features,
        (int)out_features,
        stream
    );

    return output;
}

// Operator 3: lstm_forward
std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> lstm_forward_forward(
    torch::Tensor x,
    torch::Tensor h0,
    torch::Tensor c0,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0,
    torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1,
    torch::Tensor bias_hh_l1,
    int64_t num_layers,
    int64_t hidden_size
) {
    // Ensure all tensors are contiguous float32 CUDA
    x = ensure_contiguous_float32_cuda(x, "x");
    h0 = ensure_contiguous_float32_cuda(h0, "h0");
    c0 = ensure_contiguous_float32_cuda(c0, "c0");
    weight_ih_l0 = ensure_contiguous_float32_cuda(weight_ih_l0, "weight_ih_l0");
    weight_hh_l0 = ensure_contiguous_float32_cuda(weight_hh_l0, "weight_hh_l0");
    bias_ih_l0 = ensure_contiguous_float32_cuda(bias_ih_l0, "bias_ih_l0");
    bias_hh_l0 = ensure_contiguous_float32_cuda(bias_hh_l0, "bias_hh_l0");
    weight_ih_l1 = ensure_contiguous_float32_cuda(weight_ih_l1, "weight_ih_l1");
    weight_hh_l1 = ensure_contiguous_float32_cuda(weight_hh_l1, "weight_hh_l1");
    bias_ih_l1 = ensure_contiguous_float32_cuda(bias_ih_l1, "bias_ih_l1");
    bias_hh_l1 = ensure_contiguous_float32_cuda(bias_hh_l1, "bias_hh_l1");

    // Validate shapes
    TORCH_CHECK(x.dim() == 3, "x must be 3D [seq_len, batch_size, input_size]");
    TORCH_CHECK(h0.dim() == 3, "h0 must be 3D [num_layers, batch_size, hidden_size]");
    TORCH_CHECK(c0.dim() == 3, "c0 must be 3D [num_layers, batch_size, hidden_size]");

    int64_t seq_len = x.size(0);
    int64_t batch_size = x.size(1);
    int64_t input_size = x.size(2);

    TORCH_CHECK(h0.size(0) == num_layers, "h0 layer dimension must match num_layers");
    TORCH_CHECK(h0.size(1) == batch_size, "h0 batch dimension must match x batch size");
    TORCH_CHECK(h0.size(2) == hidden_size, "h0 hidden dimension must match hidden_size");
    TORCH_CHECK(c0.size(0) == num_layers, "c0 layer dimension must match num_layers");
    TORCH_CHECK(c0.size(1) == batch_size, "c0 batch dimension must match x batch size");
    TORCH_CHECK(c0.size(2) == hidden_size, "c0 hidden dimension must match hidden_size");

    // Check weight shapes (assuming PyTorch LSTM convention: [4*hidden_size, input_size] for layer0, [4*hidden_size, hidden_size] for layer1)
    TORCH_CHECK(weight_ih_l0.dim() == 2, "weight_ih_l0 must be 2D");
    TORCH_CHECK(weight_ih_l0.size(0) == 4 * hidden_size, "weight_ih_l0 first dim must be 4*hidden_size");
    TORCH_CHECK(weight_ih_l0.size(1) == input_size, "weight_ih_l0 second dim must be input_size");
    TORCH_CHECK(weight_hh_l0.dim() == 2, "weight_hh_l0 must be 2D");
    TORCH_CHECK(weight_hh_l0.size(0) == 4 * hidden_size, "weight_hh_l0 first dim must be 4*hidden_size");
    TORCH_CHECK(weight_hh_l0.size(1) == hidden_size, "weight_hh_l0 second dim must be hidden_size");
    TORCH_CHECK(bias_ih_l0.dim() == 1 && bias_ih_l0.size(0) == 4 * hidden_size, "bias_ih_l0 size must be 4*hidden_size");
    TORCH_CHECK(bias_hh_l0.dim() == 1 && bias_hh_l0.size(0) == 4 * hidden_size, "bias_hh_l0 size must be 4*hidden_size");

    // For num_layers >= 2, check layer1 weights, but launcher only supports 1 or 2 layers.
    if (num_layers >= 2) {
        TORCH_CHECK(weight_ih_l1.dim() == 2, "weight_ih_l1 must be 2D");
        TORCH_CHECK(weight_ih_l1.size(0) == 4 * hidden_size, "weight_ih_l1 first dim must be 4*hidden_size");
        TORCH_CHECK(weight_ih_l1.size(1) == hidden_size, "weight_ih_l1 second dim must be hidden_size");
        TORCH_CHECK(weight_hh_l1.dim() == 2, "weight_hh_l1 must be 2D");
        TORCH_CHECK(weight_hh_l1.size(0) == 4 * hidden_size, "weight_hh_l1 first dim must be 4*hidden_size");
        TORCH_CHECK(weight_hh_l1.size(1) == hidden_size, "weight_hh_l1 second dim must be hidden_size");
        TORCH_CHECK(bias_ih_l1.dim() == 1 && bias_ih_l1.size(0) == 4 * hidden_size, "bias_ih_l1 size must be 4*hidden_size");
        TORCH_CHECK(bias_hh_l1.dim() == 1 && bias_hh_l1.size(0) == 4 * hidden_size, "bias_hh_l1 size must be 4*hidden_size");
    } else {
        // For num_layers < 2, we can ignore or check that the layer1 tensors are still provided but unused.
        // We'll still require them to be valid CUDA tensors but not validate their shapes strictly.
    }

    // Allocate output and final states
    auto output = torch::empty({seq_len, batch_size, hidden_size}, x.options());
    auto h_n = torch::empty({num_layers, batch_size, hidden_size}, h0.options());
    auto c_n = torch::empty({num_layers, batch_size, hidden_size}, c0.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    lstm_forward_launcher(
        x.data_ptr<float>(),
        h0.data_ptr<float>(),
        c0.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(),
        bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(),
        bias_hh_l1.data_ptr<float>(),
        output.data_ptr<float>(),
        h_n.data_ptr<float>(),
        c_n.data_ptr<float>(),
        (int)seq_len,
        (int)batch_size,
        (int)input_size,
        (int)hidden_size,
        (int)num_layers,
        stream
    );

    return std::make_tuple(output, std::make_tuple(h_n, c_n));
}

void register_custom_ops(pybind11::module& m) {
    m.def("zeros_like_alloc", &zeros_like_alloc_forward,
          py::arg("num_layers"), py::arg("batch_size"), py::arg("hidden_size"), py::arg("device"));

    m.def("lstm_forward", &lstm_forward_forward,
          py::arg("x"), py::arg("h0"), py::arg("c0"),
          py::arg("weight_ih_l0"), py::arg("weight_hh_l0"), py::arg("bias_ih_l0"), py::arg("bias_hh_l0"),
          py::arg("weight_ih_l1"), py::arg("weight_hh_l1"), py::arg("bias_ih_l1"), py::arg("bias_hh_l1"),
          py::arg("num_layers"), py::arg("hidden_size"));

    m.def("linear_fc", &linear_fc_forward,
          py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
