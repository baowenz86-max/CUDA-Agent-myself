// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream);

extern "C" void linear_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_features,
    int out_features,
    cudaStream_t stream);

// Since view_flatten is a view, we do not call any launcher.
// The placeholder launcher in .cu is not used; we directly return a view.

// -----------------------------------------------------------------------------
// adaptive_avg_pool2d_forward
// -----------------------------------------------------------------------------
torch::Tensor adaptive_avg_pool2d_forward(
    torch::Tensor x,
    int64_t output_h,
    int64_t output_w) {
    
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input x must be 4D (N, C, H, W)");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input x must be contiguous");
    TORCH_CHECK(output_h > 0 && output_w > 0, "output_h and output_w must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({N, C, output_h, output_w}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_avg_pool2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(output_h),
        static_cast<int>(output_w),
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// view_flatten_forward
// -----------------------------------------------------------------------------
torch::Tensor view_flatten_forward(
    torch::Tensor x,
    int64_t batch_size) {

    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dim() >= 1, "Input x must have at least 1 dimension");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");

    // x.view(batch_size, -1)
    auto output = x.view({batch_size, -1});
    return output;
}

// -----------------------------------------------------------------------------
// linear_forward
// -----------------------------------------------------------------------------
torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias) {

    // Input checks
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D [batch, in_features]");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D [out_features, in_features]");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");

    int batch = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "Weight in_features must match input in_features");

    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dim() == 1, "Bias must be 1D [out_features]");
        TORCH_CHECK(bias_tensor.size(0) == out_features, "Bias size must match out_features");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias must be contiguous");
    }

    auto output = torch::empty({batch, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    linear_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch,
        in_features,
        out_features,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool2d_forward", &adaptive_avg_pool2d_forward,
          "Adaptive average pooling forward",
          py::arg("x"), py::arg("output_h"), py::arg("output_w"));
    m.def("view_flatten_forward", &view_flatten_forward,
          "View flatten forward",
          py::arg("x"), py::arg("batch_size"));
    m.def("linear_forward", &linear_forward,
          "Linear forward with cuBLAS",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
