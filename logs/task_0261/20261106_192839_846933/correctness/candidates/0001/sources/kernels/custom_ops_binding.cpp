#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void one_hot_float_launcher(
    const int64_t* x_dev,
    float* out_dev,
    int64_t batch_size,
    int64_t num_classes,
    cudaStream_t stream);

extern "C" void adaptive_log_softmax_log_prob_launcher(
    const float* input,
    const float* head_weight,
    const float* head_bias,
    const float* tail0_weight0,
    const float* tail0_weight1,
    const float* tail1_weight0,
    const float* tail1_weight1,
    float* output,
    int batch_size,
    int in_features,
    int head_size,
    int tail0_hidden,
    int tail0_size,
    int tail1_hidden,
    int tail1_size,
    int n_classes,
    int cutoff0,
    int cutoff1,
    cudaStream_t stream);

// Binding for one_hot_float
torch::Tensor one_hot_float(torch::Tensor x, int64_t num_classes) {
    TORCH_CHECK(x.is_cpu(), "Input x must be a CPU tensor");
    TORCH_CHECK(x.dtype() == torch::kInt64, "Input x must be int64");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(x.dim() == 1, "Input x must be 1D");

    int64_t batch_size = x.size(0);

    // Allocate output on current CUDA device
    auto output = torch::empty({batch_size, num_classes}, x.options().dtype(torch::kFloat32).device(torch::kCUDA));

    // Copy x to device tensor (int64)
    auto x_device = x.to(torch::kCUDA);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    one_hot_float_launcher(
        x_device.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        batch_size,
        num_classes,
        stream
    );

    // Check for errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "one_hot_float kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Binding for adaptive_log_softmax_log_prob
// The cutoffs parameter is passed as a pybind object; model_new.py passes a Python list.
// We accept a pybind11::iterable and convert to std::vector<int>.
torch::Tensor adaptive_log_softmax_log_prob(
    torch::Tensor input,
    torch::Tensor head_weight,
    torch::Tensor head_bias,
    torch::Tensor tail0_weight0,
    torch::Tensor tail0_weight1,
    torch::Tensor tail1_weight0,
    torch::Tensor tail1_weight1,
    pybind11::iterable cutoffs) {

    // Check all tensors are CUDA float32 contiguous
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");

    auto check_weight = [](const torch::Tensor& w, const char* name) {
        TORCH_CHECK(w.is_cuda(), name, " must be a CUDA tensor");
        TORCH_CHECK(w.dtype() == torch::kFloat32, name, " must be float32");
        TORCH_CHECK(w.is_contiguous(), name, " must be contiguous");
        TORCH_CHECK(w.dim() == 2, name, " must be 2D");
    };
    check_weight(head_weight, "head_weight");
    check_weight(head_bias, "head_bias");
    check_weight(tail0_weight0, "tail0_weight0");
    check_weight(tail0_weight1, "tail0_weight1");
    check_weight(tail1_weight0, "tail1_weight0");
    check_weight(tail1_weight1, "tail1_weight1");

    // Extract dimensions
    int batch_size = input.size(0);
    int in_features = input.size(1);
    int head_size = head_weight.size(0);
    int tail0_hidden = tail0_weight0.size(0);
    int tail0_size = tail0_weight1.size(0);
    int tail1_hidden = tail1_weight0.size(0);
    int tail1_size = tail1_weight1.size(0);
    int n_classes = head_size + tail0_size + tail1_size;

    // Convert cutoffs iterable to vector<int>
    std::vector<int64_t> cutoff_values;
    for (auto item : cutoffs) {
        cutoff_values.push_back(pybind11::cast<int64_t>(item));
    }
    TORCH_CHECK(cutoff_values.size() == 2, "cutoffs must contain exactly 2 values");
    int cutoff0 = static_cast<int>(cutoff_values[0]);
    int cutoff1 = static_cast<int>(cutoff_values[1]);

    // Allocate output tensor [batch_size, n_classes]
    auto output = torch::empty({batch_size, n_classes}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch launcher
    adaptive_log_softmax_log_prob_launcher(
        input.data_ptr<float>(),
        head_weight.data_ptr<float>(),
        head_bias.data_ptr<float>(),
        tail0_weight0.data_ptr<float>(),
        tail0_weight1.data_ptr<float>(),
        tail1_weight0.data_ptr<float>(),
        tail1_weight1.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_features,
        head_size,
        tail0_hidden,
        tail0_size,
        tail1_hidden,
        tail1_size,
        n_classes,
        cutoff0,
        cutoff1,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "adaptive_log_softmax_log_prob kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("one_hot_float", &one_hot_float, "One-hot encode int64 CPU tensor to float32 CUDA tensor",
          pybind11::arg("x"), pybind11::arg("num_classes"));
    m.def("adaptive_log_softmax_log_prob", &adaptive_log_softmax_log_prob,
          "Compute adaptive log softmax log_prob",
          pybind11::arg("input"), pybind11::arg("head_weight"), pybind11::arg("head_bias"),
          pybind11::arg("tail0_weight0"), pybind11::arg("tail0_weight1"),
          pybind11::arg("tail1_weight0"), pybind11::arg("tail1_weight1"),
          pybind11::arg("cutoffs"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
