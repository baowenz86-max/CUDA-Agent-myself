// Binding for custom embedding and avg_pool2d CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_launcher(
    const int64_t* indices_ptr,
    const float* weight_gpu,
    float* output_gpu,
    int num_indices,
    int embedding_dim,
    cudaStream_t stream);

extern "C" void avg_pool2d_launcher(
    const float* input_gpu,
    float* output_gpu,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    cudaStream_t stream);

// Embedding forward: weight (CUDA float32 [num_embeddings, embedding_dim]), indices (int64, CPU or CUDA)
torch::Tensor embedding_forward(torch::Tensor weight, torch::Tensor indices) {
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    // Accept both CPU and CUDA indices; launcher handles pointer type

    int64_t embedding_dim = weight.size(1);
    int64_t num_indices = indices.numel();

    std::vector<int64_t> output_shape(indices.sizes().begin(), indices.sizes().end());
    output_shape.push_back(embedding_dim);
    auto output = torch::empty(output_shape, weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    embedding_launcher(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(num_indices),
        static_cast<int>(embedding_dim),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "embedding_launcher failed");

    return output;
}

// AvgPool2d forward: input (CUDA float32 [N, H, W, C]), kernel_size, stride, padding (all ints)
torch::Tensor avg_pool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, H, W, C)");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int64_t N = input.size(0);
    int64_t H = input.size(1);
    int64_t W = input.size(2);
    int64_t C = input.size(3);

    int64_t H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "invalid pooling dimensions");

    auto output = torch::empty({N, H_out, W_out, C}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    avg_pool2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(H_out),
        static_cast<int>(W_out),
        static_cast<int>(kernel_size),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(stride),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "avg_pool2d_launcher failed");

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding lookup (CPU or CUDA indices)",
          py::arg("weight"), py::arg("indices"));
    m.def("avg_pool2d_forward", &avg_pool2d_forward, "2D average pooling (CUDA)",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
