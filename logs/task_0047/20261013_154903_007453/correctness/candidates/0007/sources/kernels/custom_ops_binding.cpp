#include <torch/types.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void linear_forward(const float* input, const float* weight, const float* bias, float* output,
                    int batch, int in_features, int out_features, cudaStream_t stream);
void log_softmax_forward(const float* input, float* output,
                         int batch, int num_params, int dim, cudaStream_t stream);
void softmax_forward(const float* input, float* output,
                     int batch, int num_params, int dim, cudaStream_t stream);
void kl_div_loss_forward(const float* input, const float* target, float* output,
                         int batch, int num_params, cudaStream_t stream);
void l1_loss_forward(const float* input, const float* target, float* output,
                     int batch, cudaStream_t stream);
}

// Helper to get current CUDA stream
static inline cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Wrapper: linear_forward
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "Bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, in_features)");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D (out_features, in_features)");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    int batch = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "Weight shape mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "Bias size mismatch");

    auto output = torch::empty({batch, out_features}, input.options());

    cudaStream_t stream = getCurrentStream();
    linear_forward(input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
                   output.data_ptr<float>(), batch, in_features, out_features, stream);
    return output;
}

// Wrapper: log_softmax_forward
torch::Tensor log_softmax_forward(torch::Tensor input, int64_t dim = 1) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, num_params)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    int batch = input.size(0);
    int num_params = input.size(1);

    auto output = torch::empty_like(input);

    cudaStream_t stream = getCurrentStream();
    log_softmax_forward(input.data_ptr<float>(), output.data_ptr<float>(),
                        batch, num_params, static_cast<int>(dim), stream);
    return output;
}

// Wrapper: softmax_forward
torch::Tensor softmax_forward(torch::Tensor input, int64_t dim = 1) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, num_params)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    int batch = input.size(0);
    int num_params = input.size(1);

    auto output = torch::empty_like(input);

    cudaStream_t stream = getCurrentStream();
    softmax_forward(input.data_ptr<float>(), output.data_ptr<float>(),
                    batch, num_params, static_cast<int>(dim), stream);
    return output;
}

// Wrapper: kl_div_loss_forward
torch::Tensor kl_div_loss_forward(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "Target must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "Target must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "Target must be float32");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D (batch, num_params)");
    TORCH_CHECK(target.dim() == 2, "Target must be 2D (batch, num_params)");
    TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have same shape");

    int batch = input.size(0);
    int num_params = input.size(1);

    auto output = torch::empty({}, input.options());

    cudaStream_t stream = getCurrentStream();
    kl_div_loss_forward(input.data_ptr<float>(), target.data_ptr<float>(),
                        output.data_ptr<float>(), batch, num_params, stream);
    return output;
}

// Wrapper: l1_loss_forward
torch::Tensor l1_loss_forward(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "Target must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "Target must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "Target must be float32");
    TORCH_CHECK(input.dim() == 1, "Input must be 1D (batch_size)");
    TORCH_CHECK(target.dim() == 1, "Target must be 1D (batch_size)");
    TORCH_CHECK(input.sizes() == target.sizes(), "Input and target must have same shape");

    int batch = input.size(0);

    auto output = torch::empty({}, input.options());

    cudaStream_t stream = getCurrentStream();
    l1_loss_forward(input.data_ptr<float>(), target.data_ptr<float>(),
                    output.data_ptr<float>(), batch, stream);
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward: output = input @ weight.T + bias",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"));
    m.def("log_softmax_forward", &log_softmax_forward,
          "Log softmax over dim=1",
          pybind11::arg("input"), pybind11::arg("dim") = 1);
    m.def("softmax_forward", &softmax_forward,
          "Softmax over dim=1",
          pybind11::arg("input"), pybind11::arg("dim") = 1);
    m.def("kl_div_loss_forward", &kl_div_loss_forward,
          "KL divergence loss with batchmean reduction",
          pybind11::arg("input"), pybind11::arg("target"));
    m.def("l1_loss_forward", &l1_loss_forward,
          "L1 loss (mean absolute difference)",
          pybind11::arg("input"), pybind11::arg("target"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
