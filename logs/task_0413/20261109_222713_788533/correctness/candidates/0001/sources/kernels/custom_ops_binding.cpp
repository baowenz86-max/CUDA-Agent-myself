#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void max_dim_reduce_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int64_t dim,
    bool keepdim,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void sync_bn_inference_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    double eps,
    float* y,
    int N, int C, int H, int W,
    cudaStream_t stream);

torch::Tensor sync_bn_inference(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    // Check contiguity
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    // Check dimensions and sizes
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(weight.size(0) == C, "weight size must match C");
    TORCH_CHECK(bias.size(0) == C, "bias size must match C");
    TORCH_CHECK(running_mean.size(0) == C, "running_mean size must match C");
    TORCH_CHECK(running_var.size(0) == C, "running_var size must match C");

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sync_bn_inference_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        y.data_ptr<float>(),
        N, C, H, W,
        stream);

    return y;
}

std::tuple<torch::Tensor, torch::Tensor> max_dim_reduce(
    torch::Tensor x,
    int64_t dim,
    bool keepdim) {
    // Check device
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");

    // Check dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    // Check contiguity
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    // Check dimensions
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");

    TORCH_CHECK(dim == 1, "dim must be 1");
    TORCH_CHECK(keepdim == true, "keepdim must be true");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto values = torch::empty({N, 1, H, W}, x.options());
    auto indices = torch::empty({N, 1, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_dim_reduce_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        dim,
        keepdim,
        N, C, H, W,
        stream);

    return {values, indices};
}

void register_custom_ops(pybind11::module& m) {
    m.def("max_dim_reduce", &max_dim_reduce,
          "Max reduction over dim=1 with keepdim=True. Returns (values, indices)",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("sync_bn_inference", &sync_bn_inference,
          "SyncBatchNorm inference using running statistics",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
