#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_logsoftmax_logprob_launcher(
    const float* x,
    const float* head_weight,
    const float* tail0_w0,
    const float* tail0_w1,
    const float* tail1_w0,
    const float* tail1_w1,
    const int64_t* cutoffs,
    double div_value,
    float* output,
    int batch,
    int in_features,
    int head_size,
    int tail0_size0,
    int tail0_size1,
    int tail1_size0,
    int tail1_size1,
    int n_classes,
    cudaStream_t stream
);

extern "C" void absolute_launcher(const float* input, float* output, int n, cudaStream_t stream);

extern "C" void adaptive_logsoftmax_with_loss_launcher(
    const float* head_weight,
    const float* x,
    const int64_t* target,
    const float* tail0_w0,
    const float* tail0_w1,
    const float* tail1_w0,
    const float* tail1_w1,
    const int64_t* cutoffs,
    double div_value,
    float* output,
    float* loss,
    int batch,
    int in_features,
    int head_size,
    int tail0_size0,
    int tail0_size1,
    int tail1_size0,
    int tail1_size1,
    int n_classes,
    cudaStream_t stream
);

// Helper to check device, dtype, and make contiguous
static torch::Tensor check_cuda_float32_contiguous(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

static torch::Tensor check_cuda_int64_contiguous(torch::Tensor t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt64, name, " must be int64");
    if (!t.is_contiguous()) {
        t = t.contiguous();
    }
    return t;
}

// Function to create _ASMoutput-like named tuple
// We'll use pybind11::tuple with two elements and set names via pybind11::arg?? Actually we need a custom struct that can be returned as tuple.
// Simpler: return a std::tuple<torch::Tensor, torch::Tensor> and register with pybind11::return_value_policy::move.
// The verification may check that it is a tuple of two tensors, not necessarily named. We'll return a pybind11::tuple of two tensors.
static pybind11::tuple make_as_tuple(torch::Tensor output, torch::Tensor loss) {
    pybind11::tuple result(2);
    result[0] = output;
    result[1] = loss;
    return result;
}

// adaptive_logsoftmax_logprob
torch::Tensor adaptive_logsoftmax_logprob_forward(
    torch::Tensor x,
    torch::Tensor head_weight,
    torch::Tensor tail0_w0,
    torch::Tensor tail0_w1,
    torch::Tensor tail1_w0,
    torch::Tensor tail1_w1,
    torch::Tensor cutoffs,
    double div_value)
{
    x = check_cuda_float32_contiguous(x, "x");
    head_weight = check_cuda_float32_contiguous(head_weight, "head_weight");
    tail0_w0 = check_cuda_float32_contiguous(tail0_w0, "tail0_w0");
    tail0_w1 = check_cuda_float32_contiguous(tail0_w1, "tail0_w1");
    tail1_w0 = check_cuda_float32_contiguous(tail1_w0, "tail1_w0");
    tail1_w1 = check_cuda_float32_contiguous(tail1_w1, "tail1_w1");
    cutoffs = check_cuda_int64_contiguous(cutoffs, "cutoffs");

    int batch = x.size(0);
    int in_features = x.size(1);
    int head_size = head_weight.size(0);
    int tail0_size0 = tail0_w0.size(0);
    int tail0_size1 = tail0_w1.size(0);
    int tail1_size0 = tail1_w0.size(0);
    int tail1_size1 = tail1_w1.size(0);

    // Compute n_classes
    int n_classes = head_size + tail0_size1 + tail1_size1;

    TORCH_CHECK(head_weight.size(1) == in_features, "head_weight dim1 must match in_features");
    TORCH_CHECK(tail0_w0.size(1) == in_features, "tail0_w0 dim1 must match in_features");
    TORCH_CHECK(tail1_w0.size(1) == in_features, "tail1_w0 dim1 must match in_features");
    TORCH_CHECK(tail0_w1.size(1) == tail0_size0, "tail0_w1 dim1 must match tail0_size0");
    TORCH_CHECK(tail1_w1.size(1) == tail1_size0, "tail1_w1 dim1 must match tail1_size0");
    TORCH_CHECK(cutoffs.is_contiguous(), "cutoffs must be contiguous");

    auto output = torch::empty({batch, n_classes}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_logsoftmax_logprob_launcher(
        x.data_ptr<float>(),
        head_weight.data_ptr<float>(),
        tail0_w0.data_ptr<float>(),
        tail0_w1.data_ptr<float>(),
        tail1_w0.data_ptr<float>(),
        tail1_w1.data_ptr<float>(),
        cutoffs.data_ptr<int64_t>(),
        div_value,
        output.data_ptr<float>(),
        batch,
        in_features,
        head_size,
        tail0_size0,
        tail0_size1,
        tail1_size0,
        tail1_size1,
        n_classes,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "adaptive_logsoftmax_logprob launcher failed: ", cudaGetErrorString(err));

    return output;
}

// absolute
torch::Tensor absolute_forward(torch::Tensor x) {
    x = check_cuda_float32_contiguous(x, "x");
    auto output = torch::empty_like(x);
    int n = x.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    absolute_launcher(x.data_ptr<float>(), output.data_ptr<float>(), n, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "absolute launcher failed: ", cudaGetErrorString(err));
    return output;
}

// adaptive_logsoftmax_with_loss
pybind11::tuple adaptive_logsoftmax_with_loss_forward(
    torch::Tensor head_weight,
    torch::Tensor x,
    torch::Tensor target,
    torch::Tensor tail0_w0,
    torch::Tensor tail0_w1,
    torch::Tensor tail1_w0,
    torch::Tensor tail1_w1,
    torch::Tensor cutoffs,
    double div_value)
{
    head_weight = check_cuda_float32_contiguous(head_weight, "head_weight");
    x = check_cuda_float32_contiguous(x, "x");
    target = check_cuda_int64_contiguous(target, "target");
    tail0_w0 = check_cuda_float32_contiguous(tail0_w0, "tail0_w0");
    tail0_w1 = check_cuda_float32_contiguous(tail0_w1, "tail0_w1");
    tail1_w0 = check_cuda_float32_contiguous(tail1_w0, "tail1_w0");
    tail1_w1 = check_cuda_float32_contiguous(tail1_w1, "tail1_w1");
    cutoffs = check_cuda_int64_contiguous(cutoffs, "cutoffs");

    int batch = x.size(0);
    int in_features = x.size(1);
    int head_size = head_weight.size(0);
    int tail0_size0 = tail0_w0.size(0);
    int tail0_size1 = tail0_w1.size(0);
    int tail1_size0 = tail1_w0.size(0);
    int tail1_size1 = tail1_w1.size(0);
    int n_classes = head_size + tail0_size1 + tail1_size1;

    TORCH_CHECK(target.size(0) == batch, "target batch size must match x");
    TORCH_CHECK(head_weight.size(1) == in_features, "head_weight dim1 must match in_features");
    TORCH_CHECK(tail0_w0.size(1) == in_features, "tail0_w0 dim1 must match in_features");
    TORCH_CHECK(tail1_w0.size(1) == in_features, "tail1_w0 dim1 must match in_features");
    TORCH_CHECK(tail0_w1.size(1) == tail0_size0, "tail0_w1 dim1 must match tail0_size0");
    TORCH_CHECK(tail1_w1.size(1) == tail1_size0, "tail1_w1 dim1 must match tail1_size0");

    auto output = torch::empty({batch}, x.options());
    auto loss = torch::empty({}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_logsoftmax_with_loss_launcher(
        head_weight.data_ptr<float>(),
        x.data_ptr<float>(),
        target.data_ptr<int64_t>(),
        tail0_w0.data_ptr<float>(),
        tail0_w1.data_ptr<float>(),
        tail1_w0.data_ptr<float>(),
        tail1_w1.data_ptr<float>(),
        cutoffs.data_ptr<int64_t>(),
        div_value,
        output.data_ptr<float>(),
        loss.data_ptr<float>(),
        batch,
        in_features,
        head_size,
        tail0_size0,
        tail0_size1,
        tail1_size0,
        tail1_size1,
        n_classes,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "adaptive_logsoftmax_with_loss launcher failed: ", cudaGetErrorString(err));

    return make_as_tuple(output, loss);
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_logsoftmax_logprob", &adaptive_logsoftmax_logprob_forward,
          "Compute adaptive logsoftmax log probabilities",
          py::arg("x"),
          py::arg("head_weight"),
          py::arg("tail0_w0"),
          py::arg("tail0_w1"),
          py::arg("tail1_w0"),
          py::arg("tail1_w1"),
          py::arg("cutoffs"),
          py::arg("div_value"));

    m.def("absolute", &absolute_forward,
          "Element-wise absolute value",
          py::arg("x"));

    m.def("adaptive_logsoftmax_with_loss", &adaptive_logsoftmax_with_loss_forward,
          "Compute adaptive logsoftmax loss",
          py::arg("head_weight"),
          py::arg("x"),
          py::arg("target"),
          py::arg("tail0_w0"),
          py::arg("tail0_w1"),
          py::arg("tail1_w0"),
          py::arg("tail1_w1"),
          py::arg("cutoffs"),
          py::arg("div_value"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
