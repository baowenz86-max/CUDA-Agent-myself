#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sigmoid_forward_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w,
    cudaStream_t stream);

extern "C" void batch_norm2d_inference_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    float* y,
    int N, int C, int H, int W,
    int stride_n, int stride_c, int stride_h, int stride_w,
    cudaStream_t stream);

// PyTorch wrapper for sigmoid_forward
torch::Tensor sigmoid_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    auto y = torch::empty_like(x);

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sigmoid_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    return y;
}

// PyTorch wrapper for batch_norm2d_inference_forward
torch::Tensor batch_norm2d_inference_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == x.size(1), "weight must be 1D with size C");
    TORCH_CHECK(bias.dim() == 1 && bias.size(0) == x.size(1), "bias must be 1D with size C");
    TORCH_CHECK(running_mean.dim() == 1 && running_mean.size(0) == x.size(1), "running_mean must be 1D with size C");
    TORCH_CHECK(running_var.dim() == 1 && running_var.size(0) == x.size(1), "running_var must be 1D with size C");

    // Ensure contiguous for the 1D parameter tensors
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();
    auto running_mean_contig = running_mean.contiguous();
    auto running_var_contig = running_var.contiguous();

    auto y = torch::empty_like(x);

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm2d_inference_launcher(
        x.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        running_mean_contig.data_ptr<float>(),
        running_var_contig.data_ptr<float>(),
        static_cast<float>(eps),
        y.data_ptr<float>(),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid forward (CUDA)",
          pybind11::arg("x"));

    m.def("batch_norm2d_inference_forward", &batch_norm2d_inference_forward,
          "BatchNorm2d inference forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("eps"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
