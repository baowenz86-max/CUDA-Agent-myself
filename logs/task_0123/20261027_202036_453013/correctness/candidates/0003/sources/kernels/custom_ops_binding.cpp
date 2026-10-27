#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void exp_forward_launcher(const float* input, float* output, int n, cudaStream_t stream);
extern "C" void linear_launcher(const float* input, const float* weight, const float* bias,
                                float* output, int batch, int in_features, int out_features,
                                cudaStream_t stream);
extern "C" void lstm_launcher(const float* input,
                               const float* W_ih_l0, const float* W_hh_l0,
                               const float* b_ih_l0, const float* b_hh_l0,
                               const float* W_ih_l1, const float* W_hh_l1,
                               const float* b_ih_l1, const float* b_hh_l1,
                               float* output,
                               int seq_len, int batch, int input_size, int hidden_size,
                               cudaStream_t stream);

#define CHECK_CUDA_TENSOR(t) \
    TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor"); \
    TORCH_CHECK((t).is_contiguous(), #t " must be contiguous"); \
    TORCH_CHECK((t).dtype() == torch::kFloat32, #t " must be float32")

static inline cudaStream_t get_current_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

static inline void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

torch::Tensor lstm_forward(
    torch::Tensor input,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0,
    torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1,
    torch::Tensor bias_hh_l1,
    int64_t seq_len,
    int64_t batch_size,
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers
) {
    // Validate inputs
    CHECK_CUDA_TENSOR(input);
    CHECK_CUDA_TENSOR(weight_ih_l0);
    CHECK_CUDA_TENSOR(weight_hh_l0);
    CHECK_CUDA_TENSOR(bias_ih_l0);
    CHECK_CUDA_TENSOR(bias_hh_l0);
    CHECK_CUDA_TENSOR(weight_ih_l1);
    CHECK_CUDA_TENSOR(weight_hh_l1);
    CHECK_CUDA_TENSOR(bias_ih_l1);
    CHECK_CUDA_TENSOR(bias_hh_l1);

    TORCH_CHECK(num_layers == 2, "Only two-layer LSTM supported");
    TORCH_CHECK(input.size(0) == seq_len && input.size(1) == batch_size && input.size(2) == input_size,
                "Input shape mismatch");
    TORCH_CHECK(hidden_size > 0 && hidden_size <= 128, "hidden_size must be between 1 and 128");
    TORCH_CHECK(input_size > 0 && input_size <= 128, "input_size must be between 1 and 128");
    TORCH_CHECK(batch_size <= 64, "batch_size too large for this implementation");

    // Allocate output with torch::empty_like? Output shape is (seq_len, batch, hidden)
    auto output = torch::empty({seq_len, batch_size, hidden_size}, input.options());

    cudaStream_t stream = get_current_stream();

    lstm_launcher(
        input.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(),
        bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(),
        bias_hh_l1.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)seq_len,
        (int)batch_size,
        (int)input_size,
        (int)hidden_size,
        stream
    );
    check_cuda_error("lstm_forward kernel launch failed");
    return output;
}

torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t out_features
) {
    CHECK_CUDA_TENSOR(input);
    CHECK_CUDA_TENSOR(weight);
    CHECK_CUDA_TENSOR(bias);

    int64_t batch = input.size(0);
    int64_t in_features = input.size(1);
    TORCH_CHECK(weight.size(0) == out_features && weight.size(1) == in_features, "weight shape mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "bias size mismatch");

    auto output = torch::empty({batch, out_features}, input.options());

    cudaStream_t stream = get_current_stream();

    linear_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)batch,
        (int)in_features,
        (int)out_features,
        stream
    );
    check_cuda_error("linear_forward kernel launch failed");
    return output;
}

torch::Tensor exp_forward(torch::Tensor input) {
    CHECK_CUDA_TENSOR(input);
    auto output = torch::empty_like(input);
    cudaStream_t stream = get_current_stream();
    exp_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), (int)input.numel(), stream);
    check_cuda_error("exp_forward kernel launch failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("lstm_forward", &lstm_forward, "LSTM forward (custom CUDA kernel)",
          py::arg("input"), py::arg("weight_ih_l0"), py::arg("weight_hh_l0"),
          py::arg("bias_ih_l0"), py::arg("bias_hh_l0"),
          py::arg("weight_ih_l1"), py::arg("weight_hh_l1"),
          py::arg("bias_ih_l1"), py::arg("bias_hh_l1"),
          py::arg("seq_len"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("num_layers"));
    m.def("linear_forward", &linear_forward, "Linear forward (custom CUDA kernel)",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("out_features"));
    m.def("exp_forward", &exp_forward, "Exp forward (custom CUDA kernel)",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
