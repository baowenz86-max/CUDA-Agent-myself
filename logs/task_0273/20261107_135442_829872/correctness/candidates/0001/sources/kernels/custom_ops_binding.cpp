#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    cudaStream_t stream
);

extern "C" void pow_scalar_forward_launcher(
    const float* input,
    float* output,
    const float* exponent_ptr,
    int total_elements,
    cudaStream_t stream
);

// PyTorch wrapper for AvgPool2d forward
torch::Tensor avgpool2d_forward(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
) {
    // Validate input tensor
    TORCH_CHECK(x.is_cuda(), "avgpool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "avgpool2d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "avgpool2d_forward: input must be 4D (N,C,H,W)");
    
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Compute output spatial dimensions
    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "avgpool2d_forward: invalid output shape");

    // Allocate output tensor with same options as input
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    avgpool2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(H_out),
        static_cast<int>(W_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    return output;
}

// PyTorch wrapper for Pow scalar forward
torch::Tensor pow_scalar_forward(
    torch::Tensor x,
    torch::Tensor exponent
) {
    // Validate input tensors
    TORCH_CHECK(x.is_cuda(), "pow_scalar_forward: input x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "pow_scalar_forward: input x must be float32");
    TORCH_CHECK(exponent.is_cuda(), "pow_scalar_forward: exponent must be a CUDA tensor");
    TORCH_CHECK(exponent.dtype() == torch::kFloat32, "pow_scalar_forward: exponent must be float32");
    TORCH_CHECK(exponent.numel() == 1, "pow_scalar_forward: exponent must be a scalar (0-dim or 1-element)");

    // Ensure exponent is contiguous (0-dim tensors are contiguous by definition, but make sure)
    auto exponent_contig = exponent.contiguous();

    // Allocate output tensor with same stride as input to preserve layout
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    pow_scalar_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        exponent_contig.data_ptr<float>(),
        static_cast<int>(x.numel()),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("avgpool2d_forward", &avgpool2d_forward,
          "Average pooling 2D forward (kernel, stride, padding)",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
    m.def("pow_scalar_forward", &pow_scalar_forward,
          "Element-wise power with scalar exponent",
          py::arg("x"), py::arg("exponent"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
