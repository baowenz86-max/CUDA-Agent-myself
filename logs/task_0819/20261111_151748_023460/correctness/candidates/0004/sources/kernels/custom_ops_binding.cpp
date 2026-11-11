#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_launcher(const float* input, float* output, int n, cudaStream_t stream);
extern "C" void custom_adaptive_log_softmax_with_loss_launcher(
    const float* input,
    const int64_t* target,
    const float* head_weight,
    const float* tail0_l1_weight,
    const float* tail0_l2_weight,
    const float* tail1_l1_weight,
    const float* tail1_l2_weight,
    float* log_prob_output,
    float* loss_output,
    float* mean_loss_output,
    int batch_size,
    int in_features,
    int head_out,
    int tail0_hidden,
    int tail0_shortlist,
    int tail1_hidden,
    int tail1_shortlist,
    int cutoffs0,
    int cutoffs1,
    int n_classes,
    float div_value,
    cudaStream_t stream);

// custom_abs wrapper
torch::Tensor custom_abs(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "custom_abs: input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_abs: input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_abs_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );

    return output;
}

// custom_adaptive_log_softmax_with_loss wrapper
std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> custom_adaptive_log_softmax_with_loss(
    torch::Tensor input,
    torch::Tensor target,
    torch::Tensor head_weight,
    torch::Tensor tail0_l1_weight,
    torch::Tensor tail0_l2_weight,
    torch::Tensor tail1_l1_weight,
    torch::Tensor tail1_l2_weight,
    int64_t cutoffs,
    int64_t cutoff2,
    int64_t n_classes,
    double div_value) {

    // Checks
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(target.is_cuda(), "target must be CUDA");
    TORCH_CHECK(head_weight.is_cuda(), "head_weight must be CUDA");
    TORCH_CHECK(tail0_l1_weight.is_cuda(), "tail0_l1_weight must be CUDA");
    TORCH_CHECK(tail0_l2_weight.is_cuda(), "tail0_l2_weight must be CUDA");
    TORCH_CHECK(tail1_l1_weight.is_cuda(), "tail1_l1_weight must be CUDA");
    TORCH_CHECK(tail1_l2_weight.is_cuda(), "tail1_l2_weight must be CUDA");

    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "target must be contiguous");
    TORCH_CHECK(head_weight.is_contiguous(), "head_weight must be contiguous");
    TORCH_CHECK(tail0_l1_weight.is_contiguous(), "tail0_l1_weight must be contiguous");
    TORCH_CHECK(tail0_l2_weight.is_contiguous(), "tail0_l2_weight must be contiguous");
    TORCH_CHECK(tail1_l1_weight.is_contiguous(), "tail1_l1_weight must be contiguous");
    TORCH_CHECK(tail1_l2_weight.is_contiguous(), "tail1_l2_weight must be contiguous");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kInt64, "target must be int64");
    TORCH_CHECK(head_weight.dtype() == torch::kFloat32, "head_weight must be float32");
    TORCH_CHECK(tail0_l1_weight.dtype() == torch::kFloat32, "tail0_l1_weight must be float32");
    TORCH_CHECK(tail0_l2_weight.dtype() == torch::kFloat32, "tail0_l2_weight must be float32");
    TORCH_CHECK(tail1_l1_weight.dtype() == torch::kFloat32, "tail1_l1_weight must be float32");
    TORCH_CHECK(tail1_l2_weight.dtype() == torch::kFloat32, "tail1_l2_weight must be float32");

    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch, in_features]");
    TORCH_CHECK(target.dim() == 1, "target must be 1D");
    TORCH_CHECK(input.numel() > 0, "input must not be empty");

    int batch_size = input.size(0);
    int in_features = input.size(1);
    TORCH_CHECK(target.size(0) == batch_size, "target batch size mismatch");

    // Determine head_out and tail dimensions based on standard formula
    // head_size = cutoffs[0] + n_classes - cutoffs[-1]
    int head_out = static_cast<int>(cutoffs + n_classes - cutoff2);
    TORCH_CHECK(head_out > 0, "head_out must be positive");
    int tail0_shortlist = static_cast<int>(cutoff2 - cutoffs);
    int tail1_shortlist = static_cast<int>(n_classes - cutoff2);
    TORCH_CHECK(tail0_shortlist > 0 && tail1_shortlist > 0, "invalid cutoffs");

    int tail0_hidden = tail0_l1_weight.size(0);
    int tail1_hidden = tail1_l1_weight.size(0);

    TORCH_CHECK(head_weight.size(0) >= head_out, "head_weight rows insufficient");
    TORCH_CHECK(head_weight.size(1) == in_features, "head_weight in_features mismatch");
    TORCH_CHECK(tail0_l1_weight.size(1) == in_features, "tail0_l1_weight in_features mismatch");
    TORCH_CHECK(tail0_l2_weight.size(0) == tail0_shortlist, "tail0_l2_weight out_features mismatch");
    TORCH_CHECK(tail0_l2_weight.size(1) == tail0_hidden, "tail0_l2_weight in_features mismatch");
    TORCH_CHECK(tail1_l1_weight.size(1) == in_features, "tail1_l1_weight in_features mismatch");
    TORCH_CHECK(tail1_l2_weight.size(0) == tail1_shortlist, "tail1_l2_weight out_features mismatch");
    TORCH_CHECK(tail1_l2_weight.size(1) == tail1_hidden, "tail1_l2_weight in_features mismatch");

    TORCH_CHECK(n_classes == head_out + tail0_shortlist + tail1_shortlist, "n_classes does not match head+tails");

    // Allocate outputs
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    torch::Tensor log_prob_output = torch::empty({batch_size, static_cast<int64_t>(n_classes)}, options);
    torch::Tensor loss_output = torch::empty({batch_size}, options);
    torch::Tensor mean_loss_output = torch::empty({}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_adaptive_log_softmax_with_loss_launcher(
        input.data_ptr<float>(),
        target.data_ptr<int64_t>(),
        head_weight.data_ptr<float>(),
        tail0_l1_weight.data_ptr<float>(),
        tail0_l2_weight.data_ptr<float>(),
        tail1_l1_weight.data_ptr<float>(),
        tail1_l2_weight.data_ptr<float>(),
        log_prob_output.data_ptr<float>(),
        loss_output.data_ptr<float>(),
        mean_loss_output.data_ptr<float>(),
        batch_size,
        in_features,
        head_out,
        tail0_hidden,
        tail0_shortlist,
        tail1_hidden,
        tail1_shortlist,
        static_cast<int>(cutoffs),
        static_cast<int>(cutoff2),
        static_cast<int>(n_classes),
        static_cast<float>(div_value),
        stream
    );

    return std::make_tuple(log_prob_output, std::make_tuple(loss_output, mean_loss_output));
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "custom abs", py::arg("input"));
    m.def("custom_adaptive_log_softmax_with_loss",
          &custom_adaptive_log_softmax_with_loss,
          "custom adaptive log softmax with loss",
          py::arg("input"),
          py::arg("target"),
          py::arg("head_weight"),
          py::arg("tail0_l1_weight"),
          py::arg("tail0_l2_weight"),
          py::arg("tail1_l1_weight"),
          py::arg("tail1_l2_weight"),
          py::arg("cutoffs"),
          py::arg("cutoff2"),
          py::arg("n_classes"),
          py::arg("div_value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
