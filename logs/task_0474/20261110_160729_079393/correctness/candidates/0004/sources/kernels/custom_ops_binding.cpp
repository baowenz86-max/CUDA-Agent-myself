#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_log_softmax_launcher(
    const float* x,
    const float* head_weight,
    const float* tail_weight0,
    const float* tail_weight1,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t head_size,
    int64_t tail_hidden_size,
    int64_t tail_size,
    int64_t n_classes,
    double div_value,
    cudaStream_t stream);

extern "C" void custom_softmax_launcher(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    int64_t dim,
    cudaStream_t stream);

extern "C" void custom_clone_launcher(
    const float* input,
    float* output,
    int64_t n,
    cudaStream_t stream);

// Helper: check tensor properties
static void check_cuda_float_contiguous(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// adaptive_log_softmax binding
static torch::Tensor adaptive_log_softmax_forward(
    torch::Tensor x,
    torch::Tensor head_weight,
    torch::Tensor tail_weight0,
    torch::Tensor tail_weight1,
    int64_t cutoffs,
    double div_value) {
    
    // Check inputs
    check_cuda_float_contiguous(x, "x");
    check_cuda_float_contiguous(head_weight, "head_weight");
    check_cuda_float_contiguous(tail_weight0, "tail_weight0");
    check_cuda_float_contiguous(tail_weight1, "tail_weight1");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(head_weight.dim() == 2, "head_weight must be 2D");
    TORCH_CHECK(tail_weight0.dim() == 2, "tail_weight0 must be 2D");
    TORCH_CHECK(tail_weight1.dim() == 2, "tail_weight1 must be 2D");

    int64_t batch_size = x.size(0);
    int64_t in_features = x.size(1);
    // head_size is exactly cutoffs (not cutoffs+1) according to nn.AdaptiveLogSoftmaxWithLoss
    int64_t head_size = cutoffs;
    int64_t tail_hidden_size = tail_weight0.size(0);
    int64_t tail_size = tail_weight1.size(0);
    int64_t n_classes = head_size + tail_size;

    TORCH_CHECK(head_weight.size(0) == head_size, "head_weight first dim must equal head_size");
    TORCH_CHECK(head_weight.size(1) == in_features, "head_weight second dim must equal in_features");
    TORCH_CHECK(tail_weight0.size(1) == in_features, "tail_weight0 second dim must equal in_features");
    TORCH_CHECK(tail_weight1.size(1) == tail_hidden_size, "tail_weight1 second dim must equal tail_hidden_size");

    // Allocate output
    auto output = torch::empty({batch_size, n_classes}, x.options());

    // Get current stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    adaptive_log_softmax_launcher(
        x.data_ptr<float>(),
        head_weight.data_ptr<float>(),
        tail_weight0.data_ptr<float>(),
        tail_weight1.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        head_size,
        tail_hidden_size,
        tail_size,
        n_classes,
        div_value,
        stream
    );

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "adaptive_log_softmax launcher failed: ", cudaGetErrorString(err));

    return output;
}

// custom_softmax binding
static torch::Tensor custom_softmax_forward(
    torch::Tensor input,
    int64_t dim) {
    
    check_cuda_float_contiguous(input, "input");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");
    
    // Normalize dim
    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim == input.dim() - 1, "custom_softmax only supports last dimension");
    
    int64_t rows = 1;
    for (int64_t i = 0; i < dim; ++i) {
        rows *= input.size(i);
    }
    int64_t cols = input.size(dim);

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        dim,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax launcher failed: ", cudaGetErrorString(err));

    return output;
}

// custom_clone binding
static torch::Tensor custom_clone_forward(
    torch::Tensor input) {
    
    check_cuda_float_contiguous(input, "input");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_clone_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_clone launcher failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_log_softmax", &adaptive_log_softmax_forward,
          "AdaptiveLogSoftmaxWithLoss log_prob for one tail",
          py::arg("x"),
          py::arg("head_weight"),
          py::arg("tail_weight0"),
          py::arg("tail_weight1"),
          py::arg("cutoffs"),
          py::arg("div_value"));

    m.def("custom_softmax", &custom_softmax_forward,
          "Softmax over last dimension",
          py::arg("input"),
          py::arg("dim"));

    m.def("custom_clone", &custom_clone_forward,
          "Clone tensor",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
