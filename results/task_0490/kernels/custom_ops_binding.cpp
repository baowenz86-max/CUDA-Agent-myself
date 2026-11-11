#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(const float* x,
                                      float* y,
                                      long long numel,
                                      cudaStream_t stream);

extern "C" void instance_norm2d_no_affine_launcher(const float* x,
                                                    float* y,
                                                    int num_planes,
                                                    int plane_size,
                                                    float eps,
                                                    cudaStream_t stream);

// GELU forward wrapper
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "gelu_forward: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu_forward: input must be float32");

    auto y = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(x.data_ptr<float>(),
                          y.data_ptr<float>(),
                          x.numel(),
                          stream);
    return y;
}

// InstanceNorm2d forward wrapper (affine=False, track_running_stats=False)
torch::Tensor instance_norm2d_no_affine_forward(torch::Tensor x, double eps) {
    TORCH_CHECK(x.is_cuda(), "instance_norm2d_no_affine_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "instance_norm2d_no_affine_forward: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "instance_norm2d_no_affine_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "instance_norm2d_no_affine_forward: input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(H > 0 && W > 0, "instance_norm2d_no_affine_forward: spatial dimensions must be positive");

    int num_planes = static_cast<int>(N * C);
    int plane_size = static_cast<int>(H * W);

    auto y = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    instance_norm2d_no_affine_launcher(x.data_ptr<float>(),
                                       y.data_ptr<float>(),
                                       num_planes,
                                       plane_size,
                                       static_cast<float>(eps),
                                       stream);
    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("gelu_forward", &gelu_forward, "GELU activation forward (approx='none')",
          py::arg("x"));
    m.def("instance_norm2d_no_affine_forward", &instance_norm2d_no_affine_forward,
          "InstanceNorm2d forward (affine=False, track_running_stats=False)",
          py::arg("x"), py::arg("eps") = 1e-05);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
