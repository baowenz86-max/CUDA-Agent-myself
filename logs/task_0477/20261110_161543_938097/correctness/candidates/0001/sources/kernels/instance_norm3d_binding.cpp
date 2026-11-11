#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declared in custom_ops.cu
extern "C" void instance_norm3d_forward_launcher(
    float* output,
    const float* x,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t strideN,
    int64_t strideC,
    int64_t strideD,
    int64_t strideH,
    int64_t strideW,
    cudaStream_t stream
);

torch::Tensor instance_norm3d_forward(
    torch::Tensor x,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps = 1e-5
) {
    // Basic checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D (N,C,D,H,W)");
    TORCH_CHECK(running_mean.dim() == 1 && running_var.dim() == 1,
                "running_mean/running_var must be 1D");
    int64_t C = x.size(1);
    TORCH_CHECK(running_mean.size(0) == C, "running_mean size must match channels");
    TORCH_CHECK(running_var.size(0) == C, "running_var size must match channels");
    TORCH_CHECK(running_mean.is_contiguous(), "running_mean must be contiguous");
    TORCH_CHECK(running_var.is_contiguous(), "running_var must be contiguous");

    int64_t N = x.size(0);
    int64_t D = x.size(2);
    int64_t H = x.size(3);
    int64_t W = x.size(4);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    instance_norm3d_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        N,
        C,
        D,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        x.stride(4),
        stream
    );

    return output;
}

void register_instance_norm3d(pybind11::module& m) {
    m.def("instance_norm3d_forward", &instance_norm3d_forward,
          "InstanceNorm3d eval-mode forward",
          py::arg("x"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps") = 1e-05);
}

REGISTER_BINDING(instance_norm3d, register_instance_norm3d);
