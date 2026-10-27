#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream);

extern "C" void adaptive_log_softmax_log_prob_launcher(
    const float* input,
    const float* head_weight,
    const float* tail0_w1,
    const float* tail0_w2,
    const float* tail1_w1,
    const float* tail1_w2,
    float* output,
    int batch_size,
    int input_size,
    int head_size,
    int reduce_size0,
    int cluster_size0,
    int reduce_size1,
    int cluster_size1,
    int shortlist_size,
    float div_value,
    cudaStream_t stream);

extern "C" void topk_launcher(
    const float* input,
    float* out_values,
    int64_t* out_indices,
    int batch_size,
    int n_classes,
    int k,
    cudaStream_t stream);

// linear_forward binding
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt = c10::nullopt) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");

    // Bias optional
    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features, "bias shape must be [out_features]");
    }

    auto output = torch::empty({batch_size, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        stream
    );

    return output;
}

// adaptive_log_softmax_log_prob binding
torch::Tensor adaptive_log_softmax_log_prob(
    torch::Tensor input,
    torch::Tensor head_weight,
    torch::Tensor tail0_w1,
    torch::Tensor tail0_w2,
    torch::Tensor tail1_w1,
    torch::Tensor tail1_w2,
    c10::List<int64_t> cutoffs_list,
    double div_value) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, input_size]");
    int batch_size = input.size(0);
    int input_size = input.size(1);

    // Validate all weight tensors
    auto check_weight = [&](const torch::Tensor& w, const char* name, int expected_rows, int expected_cols) {
        TORCH_CHECK(w.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, name, " must be float32");
        TORCH_CHECK(w.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(w.dim() == 2, name, " must be 2D");
        TORCH_CHECK(w.size(0) == expected_rows && w.size(1) == expected_cols, name, " shape mismatch");
    };

    int head_size = head_weight.size(0);
    check_weight(head_weight, "head_weight", head_size, input_size);

    int reduce_size0 = tail0_w1.size(0);
    check_weight(tail0_w1, "tail0_w1", reduce_size0, input_size);
    int cluster_size0 = tail0_w2.size(0);
    check_weight(tail0_w2, "tail0_w2", cluster_size0, reduce_size0);

    int reduce_size1 = tail1_w1.size(0);
    check_weight(tail1_w1, "tail1_w1", reduce_size1, input_size);
    int cluster_size1 = tail1_w2.size(0);
    check_weight(tail1_w2, "tail1_w2", cluster_size1, reduce_size1);

    // Extract cutoffs: must be a list of at least one int, first element is shortlist_size
    TORCH_CHECK(cutoffs_list.size() >= 1, "cutoffs list must have at least one element");
    int shortlist_size = cutoffs_list[0];
    TORCH_CHECK(shortlist_size > 0, "shortlist_size must be positive");
    TORCH_CHECK(shortlist_size + cluster_size0 + cluster_size1 == head_size - cutoffs_list.size(),
                "head_size must equal shortlist_size + sum of cluster sizes + number of clusters");
    // Above check: head_size = shortlist_size + n_clusters; n_clusters = cutoffs_list.size(); so shortlist_size + cluster sums = head_size - n_clusters

    int n_classes = shortlist_size + cluster_size0 + cluster_size1;
    auto output = torch::empty({batch_size, n_classes}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    adaptive_log_softmax_log_prob_launcher(
        input.data_ptr<float>(),
        head_weight.data_ptr<float>(),
        tail0_w1.data_ptr<float>(),
        tail0_w2.data_ptr<float>(),
        tail1_w1.data_ptr<float>(),
        tail1_w2.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        input_size,
        head_size,
        reduce_size0,
        cluster_size0,
        reduce_size1,
        cluster_size1,
        shortlist_size,
        static_cast<float>(div_value),
        stream
    );

    return output;
}

// topk binding
torch::return_types::topk topk(torch::Tensor input, int64_t k) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, n_classes]");
    TORCH_CHECK(k > 0, "k must be positive");
    TORCH_CHECK(k <= input.size(1), "k must not exceed number of columns");
    TORCH_CHECK(k <= 32, "k must be <= 32 (implementation limit)");

    int batch_size = input.size(0);
    int n_classes = input.size(1);
    auto values = torch::empty({batch_size, k}, input.options());
    auto indices = torch::empty({batch_size, k}, input.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    topk_launcher(
        input.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch_size,
        n_classes,
        static_cast<int>(k),
        stream
    );

    return torch::return_types::topk(values, indices);
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward: Y = input @ weight.T + optional bias",
          py::arg("input"), py::arg("weight"), py::arg("bias") = c10::nullopt);
    m.def("adaptive_log_softmax_log_prob", &adaptive_log_softmax_log_prob,
          "Adaptive log-softmax log probabilities",
          py::arg("input"), py::arg("head_weight"), py::arg("tail0_w1"), py::arg("tail0_w2"),
          py::arg("tail1_w1"), py::arg("tail1_w2"), py::arg("cutoffs"), py::arg("div_value"));
    m.def("topk", &topk, "Top-k values and indices", py::arg("input"), py::arg("k"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
