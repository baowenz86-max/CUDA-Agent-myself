#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim1_launcher(const float* x_data,
                                          float* out_data,
                                          int N, int C, int H, int W,
                                          const long long* strides,
                                          cudaStream_t stream);

extern "C" void custom_exp_launcher(const float* x_data,
                                     float* out_data,
                                     long long numel,
                                     long long stride,
                                     cudaStream_t stream);

// Wrapper for custom_max_dim1
// Input: x (torch.Tensor) with shape [N, C, H, W] (float32, CUDA)
// Output: torch.Tensor with shape [N, H, W] (float32, CUDA, contiguous)
torch::Tensor custom_max_dim1(torch::Tensor x) {
    // Check tensor properties
    TORCH_CHECK(x.is_cuda(), "custom_max_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_max_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_max_dim1: input must be 4D (N,C,H,W)");

    // Make input contiguous for simple stride calculation.
    // The launcher accepts arbitrary strides, but we'll use contiguous layout.
    x = x.contiguous();

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Allocate output tensor of shape [N, H, W]
    auto out = torch::empty({N, H, W}, x.options());

    // Compute strides for contiguous input: element strides in each dimension
    // layout: [N, C, H, W] contiguous => strides: [C*H*W, H*W, W, 1]
    long long strides[4];
    strides[0] = (long long)C * H * W;
    strides[1] = (long long)H * W;
    strides[2] = (long long)W;
    strides[3] = 1;

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_max_dim1_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        strides,
        stream
    );

    return out;
}

// Wrapper for custom_exp
// Input: x (torch.Tensor) arbitrary shape (float32, CUDA)
// Output: torch.Tensor same shape, exp applied elementwise
torch::Tensor custom_exp(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_exp: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_exp: input must be float32");

    // Make contiguous for simple stride=1
    x = x.contiguous();
    long long numel = x.numel();
    long long stride = 1; // contiguous, so elements are consecutive

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_exp_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        numel,
        stride,
        stream
    );

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim1", &custom_max_dim1, "Custom max over dimension 1", py::arg("x"));
    m.def("custom_exp", &custom_exp, "Custom elementwise exp", py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
