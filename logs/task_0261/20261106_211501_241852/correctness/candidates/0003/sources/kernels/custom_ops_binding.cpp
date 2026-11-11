#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/ScalarType.h>
#include <ATen/cuda/CUDAContext.h>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void one_hot_forward_launcher(
    const int64_t* x,
    float* out,
    int64_t batch_size,
    int64_t num_classes,
    cudaStream_t stream);

extern "C" void adaptive_log_softmax_log_prob_forward_launcher(
    const float* x_one_hot,
    const float* head_weight,
    const float* head_bias,
    const float* tail0_0_weight,
    const float* tail0_1_weight,
    const float* tail1_0_weight,
    const float* tail1_1_weight,
    float* out,
    int64_t batch_size,
    int64_t in_features,
    int64_t head_num_classes,
    int64_t n_classes,
    int64_t cutoff0,
    int64_t cutoff1,
    float div_value,
    cudaStream_t stream);

// Helper: move tensor to CUDA if not already, and ensure contiguous
torch::Tensor ensure_cuda_contiguous(torch::Tensor t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

// one_hot_forward binding
torch::Tensor one_hot_forward(torch::Tensor x, int64_t num_classes) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "x must be int64");
    TORCH_CHECK(x.dim() == 1, "x must be 1-dimensional");
    TORCH_CHECK(num_classes > 0, "num_classes must be positive");

    x = ensure_cuda_contiguous(x, "x");
    int64_t batch_size = x.numel();

    // Allocate output: [batch_size, num_classes]
    auto out = torch::empty({batch_size, num_classes}, x.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    one_hot_forward_launcher(
        x.data_ptr<int64_t>(),
        out.data_ptr<float>(),
        batch_size,
        num_classes,
        stream
    );

    return out;
}

// adaptive_log_softmax_log_prob_forward binding
torch::Tensor adaptive_log_softmax_log_prob_forward(
    torch::Tensor x,
    torch::Tensor head_weight,
    torch::Tensor head_bias,
    torch::Tensor tail0_0_weight,
    torch::Tensor tail0_1_weight,
    torch::Tensor tail1_0_weight,
    torch::Tensor tail1_1_weight,
    py::list cutoffs,
    double div_value,
    int64_t n_classes,
    int64_t head_num_classes
) {
    // Validate x
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2-dimensional");

    // Validate weight tensors
    auto check_weight = [](const torch::Tensor& w, const std::string& name, int64_t expected_cols) {
        TORCH_CHECK(w.is_cuda(), name, " must be CUDA");
        TORCH_CHECK(w.dtype() == torch::kFloat32, name, " must be float32");
        TORCH_CHECK(w.dim() == 2, name, " must be 2D");
        TORCH_CHECK(w.size(1) == expected_cols, name, " has wrong number of columns");
    };

    int64_t in_features = x.size(1);
    check_weight(head_weight, "head_weight", in_features);
    TORCH_CHECK(head_bias.is_cuda() && head_bias.dtype() == torch::kFloat32 && head_bias.dim() == 1 && head_bias.size(0) == head_weight.size(0), "head_bias must match head_weight rows");
    check_weight(tail0_0_weight, "tail0_0_weight", in_features);
    check_weight(tail0_1_weight, "tail0_1_weight", tail0_0_weight.size(0));
    check_weight(tail1_0_weight, "tail1_0_weight", in_features);
    check_weight(tail1_1_weight, "tail1_1_weight", tail1_0_weight.size(0));

    // Validate cutoffs list: must contain exactly two integers
    TORCH_CHECK(py::len(cutoffs) == 2, "cutoffs must be a list of two integers");
    int64_t cutoff0 = py::cast<int64_t>(cutoffs[0]);
    int64_t cutoff1 = py::cast<int64_t>(cutoffs[1]);
    TORCH_CHECK(cutoff0 > 0 && cutoff1 > cutoff0, "cutoffs must be increasing and positive");

    // Ensure all inputs contiguous
    x = ensure_cuda_contiguous(x, "x");
    head_weight = ensure_cuda_contiguous(head_weight, "head_weight");
    head_bias = ensure_cuda_contiguous(head_bias, "head_bias");
    tail0_0_weight = ensure_cuda_contiguous(tail0_0_weight, "tail0_0_weight");
    tail0_1_weight = ensure_cuda_contiguous(tail0_1_weight, "tail0_1_weight");
    tail1_0_weight = ensure_cuda_contiguous(tail1_0_weight, "tail1_0_weight");
    tail1_1_weight = ensure_cuda_contiguous(tail1_1_weight, "tail1_1_weight");

    int64_t batch_size = x.size(0);

    // Allocate output: [batch_size, n_classes]
    auto out = torch::empty({batch_size, n_classes}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    adaptive_log_softmax_log_prob_forward_launcher(
        x.data_ptr<float>(),
        head_weight.data_ptr<float>(),
        head_bias.data_ptr<float>(),
        tail0_0_weight.data_ptr<float>(),
        tail0_1_weight.data_ptr<float>(),
        tail1_0_weight.data_ptr<float>(),
        tail1_1_weight.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        in_features,
        head_num_classes,
        n_classes,
        cutoff0,
        cutoff1,
        static_cast<float>(div_value),
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("one_hot_forward", &one_hot_forward,
          "one_hot_forward(x: Tensor, num_classes: int) -> Tensor",
          py::arg("x"), py::arg("num_classes"));
    m.def("adaptive_log_softmax_log_prob_forward",
          &adaptive_log_softmax_log_prob_forward,
          "adaptive_log_softmax_log_prob_forward(x, head_weight, head_bias, tail0_0_weight, tail0_1_weight, tail1_0_weight, tail1_1_weight, cutoffs, div_value, n_classes, head_num_classes) -> Tensor",
          py::arg("x"),
          py::arg("head_weight"),
          py::arg("head_bias"),
          py::arg("tail0_0_weight"),
          py::arg("tail0_1_weight"),
          py::arg("tail1_0_weight"),
          py::arg("tail1_1_weight"),
          py::arg("cutoffs"),
          py::arg("div_value"),
          py::arg("n_classes"),
          py::arg("head_num_classes"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
