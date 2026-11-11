#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softplus_forward_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream);

extern "C" void instance_norm3d_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int spatial_size,
    float eps,
    cudaStream_t stream);

// Softplus forward wrapper
torch::Tensor softplus_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure contiguous and correct layout
    auto x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softplus_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(x_contig.numel()),
        stream
    );

    return output;
}

// InstanceNorm3d forward wrapper
// Note: eps is passed as float to match the launcher and contract
torch::Tensor instance_norm3d_forward(torch::Tensor x, float eps = 1e-05f) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    // Ensure contiguous and correct layout
    auto x_contig = x.contiguous();

    int N = static_cast<int>(x_contig.size(0));
    int C = static_cast<int>(x_contig.size(1));
    int spatial_size = static_cast<int>(x_contig.numel() / (N * C));

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    instance_norm3d_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        spatial_size,
        eps,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("softplus_forward", &softplus_forward,
          "Softplus forward",
          py::arg("x"));

    m.def("instance_norm3d_forward", &instance_norm3d_forward,
          "InstanceNorm3d forward",
          py::arg("x"),
          py::arg("eps") = 1e-05f);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
