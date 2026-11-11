#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declaration of the CUDA launcher from custom_ops.cu.
// Signature matches the implementation expected in the .cu file.
extern "C" void rnn_forward_launcher(
    const float* x,            // input sequence [seq_length, batch, input_size]
    float* output,             // output [seq_length, batch, hidden_size]
    const float* weight_ih_l0, // [hidden_size, input_size]
    const float* weight_hh_l0, // [hidden_size, hidden_size]
    const float* bias_ih_l0,   // [hidden_size] or nullptr
    const float* bias_hh_l0,   // [hidden_size] or nullptr
    const float* weight_ih_l1, // [hidden_size, hidden_size]
    const float* weight_hh_l1, // [hidden_size, hidden_size]
    const float* bias_ih_l1,   // [hidden_size] or nullptr
    const float* bias_hh_l1,   // [hidden_size] or nullptr
    float* h_n,                // [num_layers, batch, hidden_size]
    int seq_length,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    bool bias,
    cudaStream_t stream
);

// PyTorch binding function
std::tuple<torch::Tensor, torch::Tensor> rnn_forward(
    torch::Tensor x,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    c10::optional<torch::Tensor> bias_ih_l0_opt,
    c10::optional<torch::Tensor> bias_hh_l0_opt,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    c10::optional<torch::Tensor> bias_ih_l1_opt,
    c10::optional<torch::Tensor> bias_hh_l1_opt,
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers,
    bool bias
) {
    // input validation
    TORCH_CHECK(x.dim() == 3, "x must be 3D tensor [seq_length, batch, input_size]");
    int64_t seq_length = x.size(0);
    int64_t batch_size = x.size(1);
    TORCH_CHECK(x.size(2) == input_size, "x.size(2) must equal input_size");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");

    // Ensure all tensors are on CUDA and float32; move them if necessary
    auto ensure_cuda_float = [](const torch::Tensor& t, const char* name) -> torch::Tensor {
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
        if (!t.is_cuda()) {
            return t.to(torch::kCUDA);
        }
        return t;
    };

    x = ensure_cuda_float(x, "x");
    weight_ih_l0 = ensure_cuda_float(weight_ih_l0, "weight_ih_l0");
    weight_hh_l0 = ensure_cuda_float(weight_hh_l0, "weight_hh_l0");
    weight_ih_l1 = ensure_cuda_float(weight_ih_l1, "weight_ih_l1");
    weight_hh_l1 = ensure_cuda_float(weight_hh_l1, "weight_hh_l1");

    torch::Tensor bias_ih_l0, bias_hh_l0, bias_ih_l1, bias_hh_l1;
    if (bias) {
        TORCH_CHECK(bias_ih_l0_opt.has_value() && bias_hh_l0_opt.has_value() &&
                    bias_ih_l1_opt.has_value() && bias_hh_l1_opt.has_value(),
                    "bias tensors must be provided when bias=True");
        bias_ih_l0 = ensure_cuda_float(bias_ih_l0_opt.value(), "bias_ih_l0");
        bias_hh_l0 = ensure_cuda_float(bias_hh_l0_opt.value(), "bias_hh_l0");
        bias_ih_l1 = ensure_cuda_float(bias_ih_l1_opt.value(), "bias_ih_l1");
        bias_hh_l1 = ensure_cuda_float(bias_hh_l1_opt.value(), "bias_hh_l1");
    }

    // allocate outputs on CUDA
    auto output = torch::empty({seq_length, batch_size, hidden_size}, x.options());
    auto h_n = torch::empty({num_layers, batch_size, hidden_size}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // call launcher
    rnn_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias ? bias_ih_l0.data_ptr<float>() : nullptr,
        bias ? bias_hh_l0.data_ptr<float>() : nullptr,
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias ? bias_ih_l1.data_ptr<float>() : nullptr,
        bias ? bias_hh_l1.data_ptr<float>() : nullptr,
        h_n.data_ptr<float>(),
        static_cast<int>(seq_length),
        static_cast<int>(batch_size),
        static_cast<int>(input_size),
        static_cast<int>(hidden_size),
        static_cast<int>(num_layers),
        bias,
        stream
    );

    // Check for CUDA errors after kernel launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(output, h_n);
}

// Registration function
void register_rnn_forward(pybind11::module& m) {
    m.def("rnn_forward", &rnn_forward,
          "RNN forward using custom CUDA kernels",
          py::arg("x"),
          py::arg("weight_ih_l0"),
          py::arg("weight_hh_l0"),
          py::arg("bias_ih_l0"),
          py::arg("bias_hh_l0"),
          py::arg("weight_ih_l1"),
          py::arg("weight_hh_l1"),
          py::arg("bias_ih_l1"),
          py::arg("bias_hh_l1"),
          py::arg("input_size"),
          py::arg("hidden_size"),
          py::arg("num_layers"),
          py::arg("bias"));
}

REGISTER_BINDING(rnn_forward, register_rnn_forward);
