#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void gemm_forward_launcher(const float* a, const float* b_t, float* out, int batch, int in_features, int out_features, cudaStream_t stream);
extern "C" void add_bias_forward_launcher(float* out, const float* gemm_ih, const float* gemm_hh, const float* bias_ih, const float* bias_hh, int batch, int hidden_size, cudaStream_t stream);
extern "C" void rnn_cell_forward_launcher(const float* input_t, const float* h_t, const float* weight_ih, const float* weight_hh, const float* bias_ih, const float* bias_hh, float* out, int batch, int input_size, int hidden_size, cudaStream_t stream);

// Helper to check CUDA tensor float32 and contiguous on CUDA
static void check_float_cuda_contiguous(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// gemm_forward: returns a @ b_t.T, shape [batch, out_features]
torch::Tensor gemm_forward(torch::Tensor a, torch::Tensor b_t) {
    check_float_cuda_contiguous(a, "a");
    check_float_cuda_contiguous(b_t, "b_t");
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b_t.dim() == 2, "b_t must be 2D");
    int batch = a.size(0);
    int in_features = a.size(1);
    int out_features = b_t.size(0);
    TORCH_CHECK(b_t.size(1) == in_features, "inner dimensions must match");
    auto out = torch::empty({batch, out_features}, a.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gemm_forward_launcher(a.data_ptr<float>(), b_t.data_ptr<float>(), out.data_ptr<float>(), batch, in_features, out_features, stream);
    return out;
}

// add_bias_forward: output = tanh(gemm_ih + gemm_hh + bias_ih + bias_hh)
torch::Tensor add_bias_forward(torch::Tensor gemm_ih, torch::Tensor gemm_hh, torch::Tensor bias_ih, torch::Tensor bias_hh) {
    check_float_cuda_contiguous(gemm_ih, "gemm_ih");
    check_float_cuda_contiguous(gemm_hh, "gemm_hh");
    check_float_cuda_contiguous(bias_ih, "bias_ih");
    check_float_cuda_contiguous(bias_hh, "bias_hh");
    TORCH_CHECK(gemm_ih.dim() == 2, "gemm_ih must be 2D");
    TORCH_CHECK(gemm_hh.dim() == 2, "gemm_hh must be 2D");
    TORCH_CHECK(bias_ih.dim() == 1, "bias_ih must be 1D");
    TORCH_CHECK(bias_hh.dim() == 1, "bias_hh must be 1D");
    int batch = gemm_ih.size(0);
    int hidden_size = gemm_ih.size(1);
    TORCH_CHECK(gemm_hh.size(0) == batch && gemm_hh.size(1) == hidden_size, "gemm_hh shape mismatch");
    TORCH_CHECK(bias_ih.size(0) == hidden_size, "bias_ih size mismatch");
    TORCH_CHECK(bias_hh.size(0) == hidden_size, "bias_hh size mismatch");
    auto out = torch::empty({batch, hidden_size}, gemm_ih.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_forward_launcher(out.data_ptr<float>(), gemm_ih.data_ptr<float>(), gemm_hh.data_ptr<float>(), bias_ih.data_ptr<float>(), bias_hh.data_ptr<float>(), batch, hidden_size, stream);
    return out;
}

// rnn_cell_forward: h_next = tanh(input_t @ weight_ih.T + h_t @ weight_hh.T + bias_ih + bias_hh)
torch::Tensor rnn_cell_forward(torch::Tensor input_t, torch::Tensor h_t, torch::Tensor weight_ih, torch::Tensor weight_hh, torch::Tensor bias_ih, torch::Tensor bias_hh) {
    check_float_cuda_contiguous(input_t, "input_t");
    check_float_cuda_contiguous(h_t, "h_t");
    check_float_cuda_contiguous(weight_ih, "weight_ih");
    check_float_cuda_contiguous(weight_hh, "weight_hh");
    check_float_cuda_contiguous(bias_ih, "bias_ih");
    check_float_cuda_contiguous(bias_hh, "bias_hh");
    TORCH_CHECK(input_t.dim() == 2, "input_t must be 2D");
    TORCH_CHECK(h_t.dim() == 2, "h_t must be 2D");
    TORCH_CHECK(weight_ih.dim() == 2, "weight_ih must be 2D");
    TORCH_CHECK(weight_hh.dim() == 2, "weight_hh must be 2D");
    TORCH_CHECK(bias_ih.dim() == 1, "bias_ih must be 1D");
    TORCH_CHECK(bias_hh.dim() == 1, "bias_hh must be 1D");
    int batch = input_t.size(0);
    int input_size = input_t.size(1);
    int hidden_size = h_t.size(1);
    TORCH_CHECK(h_t.size(0) == batch, "batch mismatch between input_t and h_t");
    TORCH_CHECK(weight_ih.size(0) == hidden_size && weight_ih.size(1) == input_size, "weight_ih shape mismatch");
    TORCH_CHECK(weight_hh.size(0) == hidden_size && weight_hh.size(1) == hidden_size, "weight_hh shape mismatch");
    TORCH_CHECK(bias_ih.size(0) == hidden_size, "bias_ih size mismatch");
    TORCH_CHECK(bias_hh.size(0) == hidden_size, "bias_hh size mismatch");
    auto out = torch::empty({batch, hidden_size}, h_t.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rnn_cell_forward_launcher(input_t.data_ptr<float>(), h_t.data_ptr<float>(), weight_ih.data_ptr<float>(), weight_hh.data_ptr<float>(), bias_ih.data_ptr<float>(), bias_hh.data_ptr<float>(), out.data_ptr<float>(), batch, input_size, hidden_size, stream);
    return out;
}

void register_rnn_binding(py::module& m) {
    m.def("gemm_forward", &gemm_forward, py::arg("a"), py::arg("b_t"));
    m.def("add_bias_forward", &add_bias_forward, py::arg("gemm_ih"), py::arg("gemm_hh"), py::arg("bias_ih"), py::arg("bias_hh"));
    m.def("rnn_cell_forward", &rnn_cell_forward, py::arg("input_t"), py::arg("h_t"), py::arg("weight_ih"), py::arg("weight_hh"), py::arg("bias_ih"), py::arg("bias_hh"));
}

REGISTER_BINDING(rnn_binding, register_rnn_binding);
