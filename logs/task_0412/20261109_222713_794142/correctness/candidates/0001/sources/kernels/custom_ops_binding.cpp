#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_batch_norm2d_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    int N, int C, int H, int W,
    float eps,
    cudaStream_t stream
);

extern "C" void custom_avg_pool2d_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int outputH, int outputW,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
);

// BatchNorm2d wrapper
torch::Tensor custom_batch_norm2d(
    torch::Tensor x,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> running_mean,
    c10::optional<torch::Tensor> running_var,
    bool training,
    double momentum = 0.1,
    double eps = 1e-05
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(!training, "training mode not supported in this binding");

    auto x_contig = x.contiguous();
    auto N = x_contig.size(0);
    auto C = x_contig.size(1);
    auto H = x_contig.size(2);
    auto W = x_contig.size(3);

    // Output same shape
    auto y = torch::empty_like(x_contig);

    // Optional tensors: ensure they are CUDA float32 contiguous if provided
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    const float* running_mean_ptr = nullptr;
    const float* running_var_ptr = nullptr;

    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda(), "weight must be CUDA");
        TORCH_CHECK(weight->dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight->numel() == C, "weight size mismatch");
        auto w_contig = weight->contiguous();
        weight_ptr = w_contig.data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->numel() == C, "bias size mismatch");
        auto b_contig = bias->contiguous();
        bias_ptr = b_contig.data_ptr<float>();
    }
    if (running_mean.has_value() && running_mean->defined()) {
        TORCH_CHECK(running_mean->is_cuda(), "running_mean must be CUDA");
        TORCH_CHECK(running_mean->dtype() == torch::kFloat32, "running_mean must be float32");
        TORCH_CHECK(running_mean->numel() == C, "running_mean size mismatch");
        auto rm_contig = running_mean->contiguous();
        running_mean_ptr = rm_contig.data_ptr<float>();
    }
    if (running_var.has_value() && running_var->defined()) {
        TORCH_CHECK(running_var->is_cuda(), "running_var must be CUDA");
        TORCH_CHECK(running_var->dtype() == torch::kFloat32, "running_var must be float32");
        TORCH_CHECK(running_var->numel() == C, "running_var size mismatch");
        auto rv_contig = running_var->contiguous();
        running_var_ptr = rv_contig.data_ptr<float>();
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_batch_norm2d_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean_ptr,
        running_var_ptr,
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<float>(eps),
        stream
    );

    // Check for errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "BatchNorm2d launcher failed: ", cudaGetErrorString(err));
    return y;
}

// AvgPool2d wrapper
torch::Tensor custom_avg_pool2d(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding = 0
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    auto x_contig = x.contiguous();
    auto N = x_contig.size(0);
    auto C = x_contig.size(1);
    auto H = x_contig.size(2);
    auto W = x_contig.size(3);

    int64_t outputH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outputW = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(outputH > 0 && outputW > 0, "Invalid output dimensions");

    auto y = torch::empty({N, C, outputH, outputW}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_avg_pool2d_launcher(
        x_contig.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(outputH),
        static_cast<int>(outputW),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "AvgPool2d launcher failed: ", cudaGetErrorString(err));
    return y;
}

// Register both functions
void register_custom_ops_binding(pybind11::module& m) {
    m.def("custom_batch_norm2d", &custom_batch_norm2d,
          "BatchNorm2d inference (evaluation mode) custom op",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("training"),
          py::arg("momentum") = 0.1,
          py::arg("eps") = 1e-05);

    m.def("custom_avg_pool2d", &custom_avg_pool2d,
          "AvgPool2d custom op",
          py::arg("x"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding") = 0);
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
