#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void conv_transpose1d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N, int C_in, int C_out, int L_in, int L_out, int kernel_size,
    int stride, int padding, int output_padding,
    cudaStream_t stream
);

torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must have shape [N, C_in, L_in]");
    TORCH_CHECK(weight.dim() == 3, "weight must have shape [C_in, C_out, kernel_size]");
    
    if (bias.has_value()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(b.dim() == 1, "bias must have shape [C_out]");
        TORCH_CHECK(b.size(0) == weight.size(1), "bias size must match C_out");
    }
    
    // Make inputs contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value().contiguous();
    }
    
    // Extract dimensions
    int64_t N = x.size(0);
    int64_t C_in = x.size(1);
    int64_t L_in = x.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);
    
    // Validate weight shape
    TORCH_CHECK(weight.size(0) == C_in, "weight shape[0] must equal C_in");
    
    // Calculate output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "calculated output length must be positive");
    
    // Allocate output tensor
    auto output = torch::empty({N, C_out, L_out}, x.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Call launcher
    conv_transpose1d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr,
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(L_in),
        static_cast<int>(L_out),
        static_cast<int>(K),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );
    
    return output;
}

void register_conv_transpose1d(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using custom CUDA kernel",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"));
}

REGISTER_BINDING(conv_transpose1d, register_conv_transpose1d);