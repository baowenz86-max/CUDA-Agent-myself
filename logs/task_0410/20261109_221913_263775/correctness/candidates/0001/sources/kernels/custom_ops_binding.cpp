#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int B,
    int T,
    int F,
    int embedding_dim,
    long long input_stride_b,
    long long input_stride_t,
    long long input_stride_f,
    long long output_stride_b,
    long long output_stride_t,
    long long output_stride_f,
    long long output_stride_e,
    long long weight_stride_0,
    long long weight_stride_1,
    cudaStream_t stream
);

extern "C" void adaptive_avg_pool2d_1x1_launcher(
    const float* input,
    float* output,
    int B,
    int T,
    int H,
    int W,
    long long input_stride_b,
    long long input_stride_t,
    long long input_stride_h,
    long long input_stride_w,
    long long output_stride_b,
    long long output_stride_t,
    cudaStream_t stream
);

// Wrapper for embedding_forward
torch::Tensor embedding_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t num_embeddings,
    int64_t embedding_dim
) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B,T,F]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");
    TORCH_CHECK(weight.size(0) == num_embeddings, "weight.size(0) must equal num_embeddings");
    TORCH_CHECK(weight.size(1) == embedding_dim, "weight.size(1) must equal embedding_dim");

    int B = input.size(0);
    int T = input.size(1);
    int F = input.size(2);

    // Allocate output [B,T,F,embedding_dim]
    auto output = torch::empty({B, T, F, embedding_dim}, input.options().dtype(torch::kFloat32));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        T,
        F,
        static_cast<int>(embedding_dim),
        input.stride(0),
        input.stride(1),
        input.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        weight.stride(0),
        weight.stride(1),
        stream
    );

    return output;
}

// Wrapper for adaptive_avg_pool2d_1x1
torch::Tensor adaptive_avg_pool2d_1x1(torch::Tensor input) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D [B,T,H,W]");

    int B = input.size(0);
    int T = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Allocate output [B,T,1,1]
    auto output = torch::empty({B, T, 1, 1}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    adaptive_avg_pool2d_1x1_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        T,
        H,
        W,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        input.stride(3),
        output.stride(0),
        output.stride(1),
        stream
    );

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward CUDA",
          py::arg("input"), py::arg("weight"), py::arg("num_embeddings"), py::arg("embedding_dim"));
    m.def("adaptive_avg_pool2d_1x1", &adaptive_avg_pool2d_1x1, "Adaptive avg pool (1,1) CUDA",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
