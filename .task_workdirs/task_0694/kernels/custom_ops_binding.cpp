#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher functions from custom_ops.cu
extern "C" void custom_sum_and_sigmoid_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    long long input_stride_N, long long input_stride_C, long long input_stride_H, long long input_stride_W,
    long long output_stride_N, long long output_stride_H, long long output_stride_W,
    int64_t total_output_elements,
    cudaStream_t stream);

extern "C" void custom_sum_only_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    long long input_stride_N, long long input_stride_C, long long input_stride_H, long long input_stride_W,
    long long output_stride_N, long long output_stride_H, long long output_stride_W,
    int64_t total_output_elements,
    cudaStream_t stream);

extern "C" void custom_sigmoid_only_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream);

// Helper to get contiguous output tensor [N, H, W] for sum operations
torch::Tensor allocate_sum_output(const torch::Tensor& x) {
    TORCH_CHECK(x.dim() == 4, "Input must be 4D for sum reduction");
    int64_t N = x.size(0);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    return torch::empty({N, H, W}, x.options().dtype(torch::kFloat32));
}

// Wrapper for fused sum and sigmoid
torch::Tensor custom_sum_and_sigmoid(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto output = allocate_sum_output(x);
    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Compute strides (note: input may be non-contiguous, so use .stride())
    long long input_stride_N = x.stride(0);
    long long input_stride_C = x.stride(1);
    long long input_stride_H = x.stride(2);
    long long input_stride_W = x.stride(3);

    // Output is contiguous: [N,H,W] -> strides [H*W, W, 1]
    long long output_stride_N = static_cast<long long>(H) * W;
    long long output_stride_H = W;
    long long output_stride_W = 1;

    int64_t total_output_elements = static_cast<int64_t>(N) * H * W;

    custom_sum_and_sigmoid_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        input_stride_N, input_stride_C, input_stride_H, input_stride_W,
        output_stride_N, output_stride_H, output_stride_W,
        total_output_elements,
        stream
    );

    return output;
}

// Wrapper for sum only
torch::Tensor custom_sum_only(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto output = allocate_sum_output(x);
    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long input_stride_N = x.stride(0);
    long long input_stride_C = x.stride(1);
    long long input_stride_H = x.stride(2);
    long long input_stride_W = x.stride(3);

    long long output_stride_N = static_cast<long long>(H) * W;
    long long output_stride_H = W;
    long long output_stride_W = 1;

    int64_t total_output_elements = static_cast<int64_t>(N) * H * W;

    custom_sum_only_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        input_stride_N, input_stride_C, input_stride_H, input_stride_W,
        output_stride_N, output_stride_H, output_stride_W,
        total_output_elements,
        stream
    );

    return output;
}

// Wrapper for sigmoid only
torch::Tensor custom_sigmoid_only(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D for sigmoid only");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous for sigmoid only");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t total_elements = x.numel();

    custom_sigmoid_only_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_sum_and_sigmoid", &custom_sum_and_sigmoid,
          "Fused sum along dimension 1 and sigmoid",
          pybind11::arg("x"), pybind11::arg("dim"));
    m.def("custom_sum_only", &custom_sum_only,
          "Sum along dimension 1 only",
          pybind11::arg("x"), pybind11::arg("dim"));
    m.def("custom_sigmoid_only", &custom_sigmoid_only,
          "Elementwise sigmoid",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);