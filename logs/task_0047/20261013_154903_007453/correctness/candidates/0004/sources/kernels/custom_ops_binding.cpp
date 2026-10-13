// Binding for custom ops implemented in custom_ops.cu
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void linear_forward_launcher_cuda(const float* input, const float* weight, const float* bias,
                                  float* output, int batch_size, int in_features, int out_features,
                                  bool has_bias, cudaStream_t stream);

void log_softmax_launcher_cuda(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream);

void softmax_launcher_cuda(const float* input, float* output, int batch_size, int num_classes, cudaStream_t stream);

void kl_div_batchmean_launcher_cuda(const float* log_probs, const float* target_probs,
                                    float* output, int batch_size, int num_classes,
                                    float* partial_sums, cudaStream_t stream);

void l1_loss_sum_params_launcher_cuda(const float* x, const float* params,
                                      float* output, int batch_size, int num_params,
                                      float* global_sum, cudaStream_t stream);
}

// ------------------------- Linear forward -------------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    int batch_size = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");

    // Input may have strides; make contiguous if needed
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    torch::Tensor bias;
    bool has_bias = false;
    const float* bias_ptr = nullptr;
    if (bias_opt.has_value() && bias_opt->defined()) {
        bias = bias_opt->contiguous();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features, "bias shape mismatch");
        has_bias = true;
        bias_ptr = bias.data_ptr<float>();
    }

    auto output = torch::empty({batch_size, out_features}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_forward_launcher_cuda(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        has_bias,
        stream
    );
    return output;
}

// ------------------------- LogSoftmax -------------------------
torch::Tensor log_softmax(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    // Contract only supports dim=1 for this 2D case
    TORCH_CHECK(dim == 1, "only dim=1 is supported");
    auto input_contig = input.contiguous();
    int batch_size = input_contig.size(0);
    int num_classes = input_contig.size(1);
    auto output = torch::empty_like(input_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_launcher_cuda(input_contig.data_ptr<float>(), output.data_ptr<float>(),
                              batch_size, num_classes, stream);
    return output;
}

// ------------------------- Softmax -------------------------
torch::Tensor softmax(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");
    auto input_contig = input.contiguous();
    int batch_size = input_contig.size(0);
    int num_classes = input_contig.size(1);
    auto output = torch::empty_like(input_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_launcher_cuda(input_contig.data_ptr<float>(), output.data_ptr<float>(),
                          batch_size, num_classes, stream);
    return output;
}

// ------------------------- KL Div batchmean -------------------------
torch::Tensor kl_div_batchmean(torch::Tensor log_probs, torch::Tensor target_probs) {
    TORCH_CHECK(log_probs.is_cuda(), "log_probs must be a CUDA tensor");
    TORCH_CHECK(target_probs.is_cuda(), "target_probs must be a CUDA tensor");
    TORCH_CHECK(log_probs.dtype() == torch::kFloat32, "log_probs must be float32");
    TORCH_CHECK(target_probs.dtype() == torch::kFloat32, "target_probs must be float32");
    TORCH_CHECK(log_probs.dim() == 2 && target_probs.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(log_probs.sizes() == target_probs.sizes(), "shape mismatch");
    auto log_probs_contig = log_probs.contiguous();
    auto target_probs_contig = target_probs.contiguous();
    int batch_size = log_probs_contig.size(0);
    int num_classes = log_probs_contig.size(1);
    auto output = torch::empty({}, log_probs_contig.options());
    // Allocate temporary buffer for per-row sums
    auto partial_sums = torch::empty({batch_size}, log_probs_contig.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    kl_div_batchmean_launcher_cuda(
        log_probs_contig.data_ptr<float>(),
        target_probs_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_classes,
        partial_sums.data_ptr<float>(),
        stream
    );
    return output;
}

// ------------------------- L1 loss sum params -------------------------
torch::Tensor l1_loss_sum_params(torch::Tensor x, torch::Tensor params) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(params.dtype() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(params.dim() == 1, "params must be 1D");
    TORCH_CHECK(x.size(1) == params.size(0), "params size mismatch");
    auto x_contig = x.contiguous();
    auto params_contig = params.contiguous();
    int batch_size = x_contig.size(0);
    int num_params = x_contig.size(1);
    auto output = torch::empty({}, x_contig.options());
    // Allocate temporary buffer for global sum (single float)
    auto global_sum = torch::empty({1}, x_contig.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    l1_loss_sum_params_launcher_cuda(
        x_contig.data_ptr<float>(),
        params_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        num_params,
        global_sum.data_ptr<float>(),
        stream
    );
    return output;
}

// ------------------------- Registration -------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward, "Linear forward using cuBLAS",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("log_softmax", &log_softmax, "LogSoftmax along dim=1",
          py::arg("input"), py::arg("dim") = 1);
    m.def("softmax", &softmax, "Softmax along dim=1",
          py::arg("input"), py::arg("dim") = 1);
    m.def("kl_div_batchmean", &kl_div_batchmean, "KL Divergence batchmean",
          py::arg("log_probs"), py::arg("target_probs"));
    m.def("l1_loss_sum_params", &l1_loss_sum_params, "Sum of L1 losses over columns",
          py::arg("x"), py::arg("params"));
}
REGISTER_BINDING(custom_ops, register_custom_ops);
