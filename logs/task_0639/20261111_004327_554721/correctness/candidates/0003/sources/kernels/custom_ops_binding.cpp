#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim1_keepdim_launcher(const float* x,
                                                float* values,
                                                int64_t* indices,
                                                int N, int C, int H, int W,
                                                int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                                cudaStream_t stream);

extern "C" void custom_layer_norm_launcher(const float* x,
                                          const float* weight,
                                          const float* bias,
                                          float* y,
                                          int N, int H, int W, int C_norm,
                                          double eps,
                                          cudaStream_t stream);

// Wrapper for custom_max_dim1_keepdim
std::tuple<torch::Tensor, torch::Tensor> custom_max_dim1_keepdim(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(x.size(1) > 0, "channel dimension must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output tensors
    auto values = torch::empty({N, 1, H, W}, x.options());
    auto indices = torch::empty({N, 1, H, W}, x.options().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launcher uses strides (even if x is non-contiguous)
    custom_max_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    return {values, indices};
}

// Wrapper for custom_layer_norm
torch::Tensor custom_layer_norm(torch::Tensor x,
                                torch::Tensor weight,
                                torch::Tensor bias,
                                double eps) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kFloat32, "weight must be CUDA float32");
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32, "bias must be CUDA float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(x.size(3) == weight.numel(), "normalized shape must match last dimension of x");
    TORCH_CHECK(weight.numel() == bias.numel(), "weight and bias must have same size");
    TORCH_CHECK(weight.is_contiguous() && bias.is_contiguous(), "weight and bias must be contiguous");

    // Ensure x is contiguous; layer_norm_kernel assumes contiguous layout
    x = x.contiguous();

    int N = x.size(0);
    int H = x.size(2);
    int W = x.size(3);
    int C_norm = W;  // last dim = normalized size

    // Output tensor (same shape as x)
    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launcher assumes x is contiguous
    custom_layer_norm_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, H, W, C_norm,
        eps,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim1_keepdim", &custom_max_dim1_keepdim,
          "Custom max over dimension 1 with keepdim=True",
          py::arg("x"));
    m.def("custom_layer_norm", &custom_layer_norm,
          "Custom layer normalization over last dimension",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-05);
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
