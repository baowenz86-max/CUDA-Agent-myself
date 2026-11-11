#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward(const float* input, float* output, int numel, cudaStream_t stream);
extern "C" void adaptive_log_softmax_loss_forward(
    const float* input,
    const int64_t* target,
    const float* head_weight,
    const float* tail0_weight0,
    const float* tail0_weight1,
    const float* tail1_weight0,
    const float* tail1_weight1,
    int batch_size,
    int in_features,
    int n_classes,
    int cutoff0,
    int cutoff1,
    int h0,
    int h1,
    float* output,
    float* loss,
    cudaStream_t stream);
extern "C" void adaptive_log_softmax_log_prob_forward(
    const float* input,
    const float* head_weight,
    const float* tail0_weight0,
    const float* tail0_weight1,
    const float* tail1_weight0,
    const float* tail1_weight1,
    int batch_size,
    int in_features,
    int n_classes,
    int cutoff0,
    int cutoff1,
    int h0,
    int h1,
    float* log_probs,
    cudaStream_t stream);

// ReLU binding
torch::Tensor relu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "relu: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "relu: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "relu: input must be contiguous");

    auto output = torch::empty_like(input);
    int numel = static_cast<int>(input.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// Adaptive log softmax loss binding
std::tuple<torch::Tensor, torch::Tensor> adaptive_log_softmax_loss(
    torch::Tensor input,
    torch::Tensor target,
    torch::Tensor head_weight,
    torch::Tensor tail0_weight0,
    torch::Tensor tail0_weight1,
    torch::Tensor tail1_weight0,
    torch::Tensor tail1_weight1,
    int64_t cutoffs,
    double div_value,
    bool head_bias) {

    // Validate devices
    TORCH_CHECK(input.is_cuda(), "adaptive_log_softmax_loss: input must be CUDA");
    TORCH_CHECK(target.is_cuda(), "adaptive_log_softmax_loss: target must be CUDA");
    TORCH_CHECK(head_weight.is_cuda(), "adaptive_log_softmax_loss: head_weight must be CUDA");
    TORCH_CHECK(tail0_weight0.is_cuda(), "adaptive_log_softmax_loss: tail0_weight0 must be CUDA");
    TORCH_CHECK(tail0_weight1.is_cuda(), "adaptive_log_softmax_loss: tail0_weight1 must be CUDA");
    TORCH_CHECK(tail1_weight0.is_cuda(), "adaptive_log_softmax_loss: tail1_weight0 must be CUDA");
    TORCH_CHECK(tail1_weight1.is_cuda(), "adaptive_log_softmax_loss: tail1_weight1 must be CUDA");

    // Validate dtypes
    TORCH_CHECK(input.dtype() == torch::kFloat32, "adaptive_log_softmax_loss: input must be float32");
    TORCH_CHECK(target.dtype() == torch::kInt64, "adaptive_log_softmax_loss: target must be int64");
    TORCH_CHECK(head_weight.dtype() == torch::kFloat32, "adaptive_log_softmax_loss: head_weight must be float32");
    TORCH_CHECK(tail0_weight0.dtype() == torch::kFloat32, "adaptive_log_softmax_loss: tail0_weight0 must be float32");
    TORCH_CHECK(tail0_weight1.dtype() == torch::kFloat32, "adaptive_log_softmax_loss: tail0_weight1 must be float32");
    TORCH_CHECK(tail1_weight0.dtype() == torch::kFloat32, "adaptive_log_softmax_loss: tail1_weight0 must be float32");
    TORCH_CHECK(tail1_weight1.dtype() == torch::kFloat32, "adaptive_log_softmax_loss: tail1_weight1 must be float32");

    // Validate contiguity
    TORCH_CHECK(input.is_contiguous(), "adaptive_log_softmax_loss: input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "adaptive_log_softmax_loss: target must be contiguous");
    TORCH_CHECK(head_weight.is_contiguous(), "adaptive_log_softmax_loss: head_weight must be contiguous");
    TORCH_CHECK(tail0_weight0.is_contiguous(), "adaptive_log_softmax_loss: tail0_weight0 must be contiguous");
    TORCH_CHECK(tail0_weight1.is_contiguous(), "adaptive_log_softmax_loss: tail0_weight1 must be contiguous");
    TORCH_CHECK(tail1_weight0.is_contiguous(), "adaptive_log_softmax_loss: tail1_weight0 must be contiguous");
    TORCH_CHECK(tail1_weight1.is_contiguous(), "adaptive_log_softmax_loss: tail1_weight1 must be contiguous");

    // Derive dimensions
    int batch_size = static_cast<int>(input.size(0));
    int in_features = static_cast<int>(input.size(1));
    int n_classes = static_cast<int>(head_weight.size(0)) - 2;  // head_weight includes two tail cluster outputs
    int cutoff0 = static_cast<int>(cutoffs);
    // cutoffs parameter is the first cutoff; we need the second cutoff as well.
    // The contract passes cutoffs as an int, but AdaptiveLogSoftmaxWithLoss expects a list of cutoffs.
    // We need to derive cutoff1 from the model's cutoffs. Since model_new.py passes self.cutoffs[0] as cutoffs,
    // we can't get cutoff1 here unless we recompute from weights or the contract. However, the launcher requires cutoff1.
    // In the .cu launcher signature, cutoff1 is needed. We need to compute it from the tail0_weight1 shape or use a convention.
    // The tail0_weight1 shape is [cutoff1 - cutoff0, h0]. So cutoff1 = cutoff0 + tail0_weight1.size(0).
    int cutoff1 = cutoff0 + static_cast<int>(tail0_weight1.size(0));
    int h0 = static_cast<int>(tail0_weight0.size(0));
    int h1 = static_cast<int>(tail1_weight0.size(0));

    TORCH_CHECK(cutoff0 > 0 && cutoff1 > cutoff0, "adaptive_log_softmax_loss: invalid cutoffs");
    TORCH_CHECK(n_classes == cutoff1 + tail1_weight1.size(0), "adaptive_log_softmax_loss: n_classes mismatch");

    // Allocate output and loss
    auto output = torch::empty({batch_size}, input.options());
    auto loss = torch::empty({}, input.options());  // scalar

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_log_softmax_loss_forward(
        input.data_ptr<float>(),
        target.data_ptr<int64_t>(),
        head_weight.data_ptr<float>(),
        tail0_weight0.data_ptr<float>(),
        tail0_weight1.data_ptr<float>(),
        tail1_weight0.data_ptr<float>(),
        tail1_weight1.data_ptr<float>(),
        batch_size,
        in_features,
        n_classes,
        cutoff0,
        cutoff1,
        h0,
        h1,
        output.data_ptr<float>(),
        loss.data_ptr<float>(),
        stream
    );

    return std::make_tuple(output, loss);
}

// Adaptive log softmax log_prob binding
torch::Tensor adaptive_log_softmax_log_prob(
    torch::Tensor input,
    torch::Tensor head_weight,
    torch::Tensor tail0_weight0,
    torch::Tensor tail0_weight1,
    torch::Tensor tail1_weight0,
    torch::Tensor tail1_weight1,
    int64_t cutoffs,
    double div_value,
    bool head_bias) {

    // Validate devices
    TORCH_CHECK(input.is_cuda(), "adaptive_log_softmax_log_prob: input must be CUDA");
    TORCH_CHECK(head_weight.is_cuda(), "adaptive_log_softmax_log_prob: head_weight must be CUDA");
    TORCH_CHECK(tail0_weight0.is_cuda(), "adaptive_log_softmax_log_prob: tail0_weight0 must be CUDA");
    TORCH_CHECK(tail0_weight1.is_cuda(), "adaptive_log_softmax_log_prob: tail0_weight1 must be CUDA");
    TORCH_CHECK(tail1_weight0.is_cuda(), "adaptive_log_softmax_log_prob: tail1_weight0 must be CUDA");
    TORCH_CHECK(tail1_weight1.is_cuda(), "adaptive_log_softmax_log_prob: tail1_weight1 must be CUDA");

    // Validate dtypes
    TORCH_CHECK(input.dtype() == torch::kFloat32, "adaptive_log_softmax_log_prob: input must be float32");
    TORCH_CHECK(head_weight.dtype() == torch::kFloat32, "adaptive_log_softmax_log_prob: head_weight must be float32");
    TORCH_CHECK(tail0_weight0.dtype() == torch::kFloat32, "adaptive_log_softmax_log_prob: tail0_weight0 must be float32");
    TORCH_CHECK(tail0_weight1.dtype() == torch::kFloat32, "adaptive_log_softmax_log_prob: tail0_weight1 must be float32");
    TORCH_CHECK(tail1_weight0.dtype() == torch::kFloat32, "adaptive_log_softmax_log_prob: tail1_weight0 must be float32");
    TORCH_CHECK(tail1_weight1.dtype() == torch::kFloat32, "adaptive_log_softmax_log_prob: tail1_weight1 must be float32");

    // Validate contiguity
    TORCH_CHECK(input.is_contiguous(), "adaptive_log_softmax_log_prob: input must be contiguous");
    TORCH_CHECK(head_weight.is_contiguous(), "adaptive_log_softmax_log_prob: head_weight must be contiguous");
    TORCH_CHECK(tail0_weight0.is_contiguous(), "adaptive_log_softmax_log_prob: tail0_weight0 must be contiguous");
    TORCH_CHECK(tail0_weight1.is_contiguous(), "adaptive_log_softmax_log_prob: tail0_weight1 must be contiguous");
    TORCH_CHECK(tail1_weight0.is_contiguous(), "adaptive_log_softmax_log_prob: tail1_weight0 must be contiguous");
    TORCH_CHECK(tail1_weight1.is_contiguous(), "adaptive_log_softmax_log_prob: tail1_weight1 must be contiguous");

    // Derive dimensions
    int batch_size = static_cast<int>(input.size(0));
    int in_features = static_cast<int>(input.size(1));
    int n_classes = static_cast<int>(head_weight.size(0)) - 2;
    int cutoff0 = static_cast<int>(cutoffs);
    int cutoff1 = cutoff0 + static_cast<int>(tail0_weight1.size(0));
    int h0 = static_cast<int>(tail0_weight0.size(0));
    int h1 = static_cast<int>(tail1_weight0.size(0));

    TORCH_CHECK(cutoff0 > 0 && cutoff1 > cutoff0, "adaptive_log_softmax_log_prob: invalid cutoffs");
    TORCH_CHECK(n_classes == cutoff1 + tail1_weight1.size(0), "adaptive_log_softmax_log_prob: n_classes mismatch");

    // Allocate output tensor of shape [batch_size, n_classes]
    auto log_probs = torch::empty({batch_size, n_classes}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_log_softmax_log_prob_forward(
        input.data_ptr<float>(),
        head_weight.data_ptr<float>(),
        tail0_weight0.data_ptr<float>(),
        tail0_weight1.data_ptr<float>(),
        tail1_weight0.data_ptr<float>(),
        tail1_weight1.data_ptr<float>(),
        batch_size,
        in_features,
        n_classes,
        cutoff0,
        cutoff1,
        h0,
        h1,
        log_probs.data_ptr<float>(),
        stream
    );

    return log_probs;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("relu", &relu, "ReLU activation",
          py::arg("input"));
    m.def("adaptive_log_softmax_loss", &adaptive_log_softmax_loss,
          "Adaptive log softmax loss forward",
          py::arg("input"),
          py::arg("target"),
          py::arg("head_weight"),
          py::arg("tail0_weight0"),
          py::arg("tail0_weight1"),
          py::arg("tail1_weight0"),
          py::arg("tail1_weight1"),
          py::arg("cutoffs"),
          py::arg("div_value"),
          py::arg("head_bias"));
    m.def("adaptive_log_softmax_log_prob", &adaptive_log_softmax_log_prob,
          "Adaptive log softmax log_prob forward",
          py::arg("input"),
          py::arg("head_weight"),
          py::arg("tail0_weight0"),
          py::arg("tail0_weight1"),
          py::arg("tail1_weight0"),
          py::arg("tail1_weight1"),
          py::arg("cutoffs"),
          py::arg("div_value"),
          py::arg("head_bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
