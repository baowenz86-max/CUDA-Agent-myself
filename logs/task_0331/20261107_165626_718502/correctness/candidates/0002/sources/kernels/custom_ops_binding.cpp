#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linear_forward_launcher(const float* x,
                                         const float* weight,
                                         const float* bias,
                                         float* y,
                                         int batch,
                                         int in_features,
                                         cudaStream_t stream);

extern "C" void gelu_forward_launcher(const float* x,
                                       float* y,
                                       int n,
                                       cudaStream_t stream);

extern "C" void gt_scalar_forward_launcher(const float* x,
                                            float threshold,
                                            bool* y,
                                            int n,
                                            cudaStream_t stream);

extern "C" void bool_add_bias_forward_launcher(const bool* x,
                                               const float* bias,
                                               float* out,
                                               int batch,
                                               int in_features,
                                               cudaStream_t stream);

// Helper to check CUDA errors after kernel launch
static void checkCudaError() {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}

// -----------------------------------------------------------------------------
// Linear forward: y = x @ weight.T + bias
// -----------------------------------------------------------------------------
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    // Validate device
    TORCH_CHECK(x.is_cuda(), "linear_forward: x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "linear_forward: weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "linear_forward: bias must be a CUDA tensor");
    // Validate dtype
    TORCH_CHECK(x.dtype() == torch::kFloat32, "linear_forward: x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "linear_forward: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "linear_forward: bias must be float32");
    // Validate contiguity
    TORCH_CHECK(x.is_contiguous(), "linear_forward: x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "linear_forward: weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "linear_forward: bias must be contiguous");
    // Validate shapes
    TORCH_CHECK(x.dim() == 2, "linear_forward: x must be 2D [batch, in_features]");
    TORCH_CHECK(weight.dim() == 2, "linear_forward: weight must be 2D [in_features, in_features]");
    TORCH_CHECK(bias.dim() == 1, "linear_forward: bias must be 1D [in_features]");
    int64_t batch = x.size(0);
    int64_t in_features = x.size(1);
    TORCH_CHECK(weight.size(0) == in_features && weight.size(1) == in_features,
                "linear_forward: weight shape must be [in_features, in_features]");
    TORCH_CHECK(bias.size(0) == in_features, "linear_forward: bias size must match in_features");

    auto y = torch::empty({batch, in_features}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_forward_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_features),
        stream
    );
    checkCudaError();
    return y;
}

// -----------------------------------------------------------------------------
// GELU forward (default approximate='none')
// -----------------------------------------------------------------------------
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu_forward: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "gelu_forward: x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "gelu_forward: x must be 2D [batch, in_features]");

    auto y = torch::empty_like(x);
    int64_t n = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_forward_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<int>(n),
        stream
    );
    checkCudaError();
    return y;
}

// -----------------------------------------------------------------------------
// Greater-than scalar: y = (x > threshold) as bool tensor
// -----------------------------------------------------------------------------
torch::Tensor gt_scalar_forward(torch::Tensor x, double threshold) {
    TORCH_CHECK(x.is_cuda(), "gt_scalar_forward: x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gt_scalar_forward: x must be float32");
    TORCH_CHECK(x.is_contiguous(), "gt_scalar_forward: x must be contiguous");
    TORCH_CHECK(x.dim() == 2, "gt_scalar_forward: x must be 2D [batch, in_features]");

    auto y = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    int64_t n = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gt_scalar_forward_launcher(
        x.data_ptr<float>(),
        static_cast<float>(threshold),
        y.data_ptr<bool>(),
        static_cast<int>(n),
        stream
    );
    checkCudaError();
    return y;
}

// -----------------------------------------------------------------------------
// Bool add bias: x bool [batch, in_features], bias float [in_features,1,1]
// Output: float [in_features, batch, in_features]
// -----------------------------------------------------------------------------
torch::Tensor bool_add_bias_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "bool_add_bias_forward: x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bool_add_bias_forward: bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kBool, "bool_add_bias_forward: x must be bool");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bool_add_bias_forward: bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "bool_add_bias_forward: x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bool_add_bias_forward: bias must be contiguous");
    TORCH_CHECK(x.dim() == 2, "bool_add_bias_forward: x must be 2D [batch, in_features]");
    TORCH_CHECK(bias.dim() == 3 && bias.size(1) == 1 && bias.size(2) == 1,
                "bool_add_bias_forward: bias must be 3D [in_features, 1, 1]");
    int64_t batch = x.size(0);
    int64_t in_features = x.size(1);
    int64_t bias_in_features = bias.size(0);
    TORCH_CHECK(bias_in_features == in_features,
                "bool_add_bias_forward: bias first dim must match x second dim");

    auto out = torch::empty({bias_in_features, batch, in_features}, x.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bool_add_bias_forward_launcher(
        x.data_ptr<bool>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_features),
        stream
    );
    checkCudaError();
    return out;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward, "Linear forward (CUDA)",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"));
    m.def("gelu_forward", &gelu_forward, "GELU forward (CUDA)",
          pybind11::arg("x"));
    m.def("gt_scalar_forward", &gt_scalar_forward, "Greater-than scalar forward (CUDA)",
          pybind11::arg("x"), pybind11::arg("threshold"));
    m.def("bool_add_bias_forward", &bool_add_bias_forward, "Bool add bias forward (CUDA)",
          pybind11::arg("x"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
