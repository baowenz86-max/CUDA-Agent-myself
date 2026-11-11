#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void lstm_forward_launcher(
    const float* x,
    const float* weight_ih_l0,
    const float* weight_hh_l0,
    const float* bias_ih_l0,
    const float* bias_hh_l0,
    const float* weight_ih_l1,
    const float* weight_hh_l1,
    const float* bias_ih_l1,
    const float* bias_hh_l1,
    int64_t hidden_size,
    int64_t num_layers,
    float* output,
    float* hidden_out,
    float* cell_out,
    int64_t batch_size,
    int64_t seq_len,
    int64_t input_size,
    cudaStream_t stream
);

extern "C" void cat_hidden_cell_launcher(
    const float* hidden,
    const float* cell,
    float* output,
    int64_t num_layers,
    int64_t batch_size,
    int64_t hidden_size,
    cudaStream_t stream
);

static void check_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstm_forward_wrapper(
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
    int64_t num_layers
) {
    // Check all tensors
    check_cuda_float_contiguous(x, "x");
    check_cuda_float_contiguous(weight_ih_l0, "weight_ih_l0");
    check_cuda_float_contiguous(weight_hh_l0, "weight_hh_l0");
    check_cuda_float_contiguous(bias_ih_l0, "bias_ih_l0");
    check_cuda_float_contiguous(bias_hh_l0, "bias_hh_l0");
    check_cuda_float_contiguous(weight_ih_l1, "weight_ih_l1");
    check_cuda_float_contiguous(weight_hh_l1, "weight_hh_l1");
    check_cuda_float_contiguous(bias_ih_l1, "bias_ih_l1");
    check_cuda_float_contiguous(bias_hh_l1, "bias_hh_l1");

    // Extract dimensions from x (batch, seq, input) and hidden_size/num_layers
    int64_t batch_size = x.size(0);
    int64_t seq_len = x.size(1);
    int64_t input_size = x.size(2);

    TORCH_CHECK(batch_size > 0 && seq_len > 0 && input_size > 0, "Invalid input dimensions");
    TORCH_CHECK(hidden_size > 0 && num_layers > 0, "hidden_size and num_layers must be positive");

    // Validate parameter shapes
    TORCH_CHECK(weight_ih_l0.size(0) == 4 * hidden_size && weight_ih_l0.size(1) == input_size,
                "weight_ih_l0 shape mismatch");
    TORCH_CHECK(weight_hh_l0.size(0) == 4 * hidden_size && weight_hh_l0.size(1) == hidden_size,
                "weight_hh_l0 shape mismatch");
    TORCH_CHECK(bias_ih_l0.size(0) == 4 * hidden_size, "bias_ih_l0 shape mismatch");
    TORCH_CHECK(bias_hh_l0.size(0) == 4 * hidden_size, "bias_hh_l0 shape mismatch");
    TORCH_CHECK(weight_ih_l1.size(0) == 4 * hidden_size && weight_ih_l1.size(1) == hidden_size,
                "weight_ih_l1 shape mismatch");
    TORCH_CHECK(weight_hh_l1.size(0) == 4 * hidden_size && weight_hh_l1.size(1) == hidden_size,
                "weight_hh_l1 shape mismatch");
    TORCH_CHECK(bias_ih_l1.size(0) == 4 * hidden_size, "bias_ih_l1 shape mismatch");
    TORCH_CHECK(bias_hh_l1.size(0) == 4 * hidden_size, "bias_hh_l1 shape mismatch");

    // Allocate output tensors
    auto output = torch::empty({batch_size, seq_len, hidden_size}, x.options());
    auto hidden = torch::empty({num_layers, batch_size, hidden_size}, x.options());
    auto cell = torch::empty({num_layers, batch_size, hidden_size}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    lstm_forward_launcher(
        x.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(),
        bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(),
        bias_hh_l1.data_ptr<float>(),
        hidden_size,
        num_layers,
        output.data_ptr<float>(),
        hidden.data_ptr<float>(),
        cell.data_ptr<float>(),
        batch_size,
        seq_len,
        input_size,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in lstm_forward: ", cudaGetErrorString(err));

    return std::make_tuple(output, hidden, cell);
}

torch::Tensor cat_hidden_cell_wrapper(torch::Tensor hidden, torch::Tensor cell) {
    check_cuda_float_contiguous(hidden, "hidden");
    check_cuda_float_contiguous(cell, "cell");

    int64_t num_layers = hidden.size(0);
    int64_t batch_size = hidden.size(1);
    int64_t hidden_size = hidden.size(2);

    TORCH_CHECK(cell.size(0) == num_layers && cell.size(1) == batch_size && cell.size(2) == hidden_size,
                "cell shape must match hidden shape");

    auto output = torch::empty({num_layers, 2 * batch_size, hidden_size}, hidden.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cat_hidden_cell_launcher(
        hidden.data_ptr<float>(),
        cell.data_ptr<float>(),
        output.data_ptr<float>(),
        num_layers,
        batch_size,
        hidden_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in cat_hidden_cell: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("lstm_forward", &lstm_forward_wrapper,
          py::arg("x"),
          py::arg("weight_ih_l0"),
          py::arg("weight_hh_l0"),
          py::arg("bias_ih_l0"),
          py::arg("bias_hh_l0"),
          py::arg("weight_ih_l1"),
          py::arg("weight_hh_l1"),
          py::arg("bias_ih_l1"),
          py::arg("bias_hh_l1"),
          py::arg("hidden_size"),
          py::arg("num_layers"));

    m.def("cat_hidden_cell", &cat_hidden_cell_wrapper,
          py::arg("hidden"),
          py::arg("cell"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
