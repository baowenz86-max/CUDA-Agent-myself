#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" void launch_custom_lstm_forward(
    const float* input,
    float* output,
    float* hn,
    float* cn,
    float* temp_output,
    const float* W_ih_l0,
    const float* W_hh_l0,
    const float* b_ih_l0,
    const float* b_hh_l0,
    const float* W_ih_l1,
    const float* W_hh_l1,
    const float* b_ih_l1,
    const float* b_hh_l1,
    int batch,
    int seq_len,
    int input_size,
    int hidden_size,
    cudaStream_t stream
);

extern "C" void custom_isin_launcher(
    const float* input,
    const int64_t* values,
    bool* mask,
    int num_values,
    int64_t total_elements,
    cudaStream_t stream
);

// Helper to check CUDA, contiguous, dtype float32
static void check_float32_cuda_contig(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// custom_lstm_forward wrapper
pybind11::object custom_lstm_forward(
    torch::Tensor input,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0,
    torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1,
    torch::Tensor bias_hh_l1,
    int64_t num_layers,
    int64_t hidden_size,
    int64_t input_size
) {
    // Validate num_layers, hidden_size, input_size
    TORCH_CHECK(num_layers == 2, "Only num_layers=2 is supported");
    TORCH_CHECK(hidden_size == 32, "Only hidden_size=32 is supported");
    TORCH_CHECK(input_size == 16, "Only input_size=16 is supported");

    // Validate input and weight tensors
    check_float32_cuda_contig(input, "input");
    check_float32_cuda_contig(weight_ih_l0, "weight_ih_l0");
    check_float32_cuda_contig(weight_hh_l0, "weight_hh_l0");
    check_float32_cuda_contig(bias_ih_l0, "bias_ih_l0");
    check_float32_cuda_contig(bias_hh_l0, "bias_hh_l0");
    check_float32_cuda_contig(weight_ih_l1, "weight_ih_l1");
    check_float32_cuda_contig(weight_hh_l1, "weight_hh_l1");
    check_float32_cuda_contig(bias_ih_l1, "bias_ih_l1");
    check_float32_cuda_contig(bias_hh_l1, "bias_hh_l1");

    // Extract sizes
    int batch = input.size(0);
    int seq_len = input.size(1);
    int in_size = input.size(2);
    TORCH_CHECK(in_size == input_size, "input_size mismatch");

    // Allocate output, hn, cn, temp
    auto output = torch::empty({batch, seq_len, hidden_size}, input.options());
    auto hn = torch::empty({num_layers, batch, hidden_size}, input.options());
    auto cn = torch::empty({num_layers, batch, hidden_size}, input.options());
    auto temp_output = torch::empty({batch, seq_len, hidden_size}, input.options());

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    launch_custom_lstm_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        hn.data_ptr<float>(),
        cn.data_ptr<float>(),
        temp_output.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(),
        bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(),
        bias_hh_l1.data_ptr<float>(),
        batch,
        seq_len,
        input_size,
        hidden_size,
        stream
    );

    // Check for launch errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in custom_lstm_forward: ", cudaGetErrorString(err));

    return pybind11::make_tuple(output, hn, cn);
}

// custom_isin_forward wrapper
torch::Tensor custom_isin_forward(torch::Tensor input, torch::Tensor values) {
    // Validate input
    check_float32_cuda_contig(input, "input");

    // Validate values
    TORCH_CHECK(values.is_cuda(), "values must be a CUDA tensor");
    TORCH_CHECK(values.is_contiguous(), "values must be contiguous");
    TORCH_CHECK(values.dtype() == torch::kInt64, "values must be int64");

    int64_t total_elements = input.numel();
    int num_values = values.numel();

    // Allocate bool mask
    auto mask = torch::empty(input.sizes(), input.options().dtype(torch::kBool));

    // Get CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_isin_launcher(
        input.data_ptr<float>(),
        values.data_ptr<int64_t>(),
        mask.data_ptr<bool>(),
        num_values,
        total_elements,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in custom_isin_forward: ", cudaGetErrorString(err));

    return mask;
}

// Registration
void register_custom_lstm_forward(pybind11::module& m) {
    m.def("custom_lstm_forward", &custom_lstm_forward,
          "Custom 2-layer LSTM forward",
          py::arg("input"),
          py::arg("weight_ih_l0"),
          py::arg("weight_hh_l0"),
          py::arg("bias_ih_l0"),
          py::arg("bias_hh_l0"),
          py::arg("weight_ih_l1"),
          py::arg("weight_hh_l1"),
          py::arg("bias_ih_l1"),
          py::arg("bias_hh_l1"),
          py::arg("num_layers"),
          py::arg("hidden_size"),
          py::arg("input_size"));
}

void register_custom_isin_forward(pybind11::module& m) {
    m.def("custom_isin_forward", &custom_isin_forward,
          "Custom isin forward",
          py::arg("input"),
          py::arg("values"));
}

REGISTER_BINDING(custom_lstm_forward, register_custom_lstm_forward);
REGISTER_BINDING(custom_isin_forward, register_custom_isin_forward);
