#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void avg_pool2d_launcher(
    float* output,
    const float* input,
    int B, int C, int H, int W,
    int kernel_size, int stride, int padding,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_height_stride,
    long long input_width_stride,
    cudaStream_t stream);

extern "C" void min_dim_launcher(
    float* output,
    const float* input,
    int B, int C, int H, int W,
    int dim,
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_height_stride,
    long long input_width_stride,
    cudaStream_t stream);

// Wrapper for avg_pool2d_forward
torch::Tensor avg_pool2d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Validate kernel_size, stride, padding
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int outH = (H + 2 * padding - kernel_size) / stride + 1;
    int outW = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "Invalid output size");

    auto output = torch::empty({B, C, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avg_pool2d_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        B, C, H, W,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for min_dim_forward
torch::Tensor min_dim_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (B, C, H, W)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty({B, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    min_dim_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        B, C, H, W,
        static_cast<int>(dim),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("avg_pool2d_forward", &avg_pool2d_forward,
          "Average pooling 2D forward",
          py::arg("x"),
          py::arg("kernel_size"),
          py::arg("stride"),
          py::arg("padding"));

    m.def("min_dim_forward", &min_dim_forward,
          "Minimum over dimension 1",
          py::arg("x"),
          py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
