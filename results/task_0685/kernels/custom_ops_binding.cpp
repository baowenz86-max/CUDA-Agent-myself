#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim0_keepdim_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    cudaStream_t stream);

extern "C" void group_norm_launcher(
    const float* x,
    float* y,
    int N, int C, int H, int W,
    int num_groups,
    float eps,
    float* mean_out,
    float* invstd_out,
    cudaStream_t stream);

// Wrapper for mean over dimension 0 with keepdim=True
torch::Tensor mean_dim0_keepdim_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output shape: [1, C, H, W], contiguous
    auto out = torch::empty({1, C, H, W}, x.options());

    // Get strides (long long for launcher)
    long long sx_b = x.stride(0);
    long long sx_c = x.stride(1);
    long long sx_h = x.stride(2);
    long long sx_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim0_keepdim_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        sx_b, sx_c, sx_h, sx_w,
        stream);

    return out;
}

// Wrapper for group normalization (no affine parameters)
torch::Tensor group_norm_forward(torch::Tensor x, int64_t num_groups) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(num_groups > 0, "num_groups must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(C % num_groups == 0, "num_groups must divide number of channels");

    // Output same shape as input (contiguous)
    auto y = torch::empty_like(x);

    // Temporary storage for mean and invstd per group
    auto mean = torch::empty({num_groups}, x.options());
    auto invstd = torch::empty({num_groups}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    float eps = 1e-5f;
    group_norm_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(num_groups),
        eps,
        mean.data_ptr<float>(),
        invstd.data_ptr<float>(),
        stream);

    return y;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim0_keepdim_forward", &mean_dim0_keepdim_forward, 
          "Mean over dimension 0 with keepdim=True",
          py::arg("x"));
    m.def("group_norm_forward", &group_norm_forward,
          "Group normalization (affine=False, eps=1e-5)",
          py::arg("x"), py::arg("num_groups"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
