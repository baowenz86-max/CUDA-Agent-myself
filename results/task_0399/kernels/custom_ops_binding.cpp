// Custom ops binding for embedding and gather
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void custom_embedding_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int64_t B,
    int64_t T,
    int64_t E,
    int64_t num_embeddings,
    cudaStream_t stream);

extern "C" void custom_gather_launcher(
    const float* input,
    const int64_t* index,
    float* output,
    int64_t B,
    int64_t T,
    int64_t I,
    int64_t E,
    int64_t input_stride_b,
    int64_t input_stride_t,
    int64_t input_stride_e,
    int64_t index_stride_b,
    int64_t index_stride_i,
    int64_t index_stride_e,
    int64_t dim,
    cudaStream_t stream);

// Custom embedding: output[b,t,:] = weight[input[b,t],:]
torch::Tensor custom_embedding(torch::Tensor input, torch::Tensor weight) {
    // Check device and dtype
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    int64_t B = input.size(0);
    int64_t T = input.size(1);
    int64_t num_embeddings = weight.size(0);
    int64_t E = weight.size(1);

    // Allocate output [B, T, E]
    auto output = torch::empty({B, T, E}, input.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_embedding_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, T, E, num_embeddings,
        stream
    );

    return output;
}

// Custom gather along dim=1: output[b,i,e] = input[b, index[b,i,e], e]
torch::Tensor custom_gather(torch::Tensor input, torch::Tensor index, int64_t dim = 1) {
    // Check device and dtype
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(index.is_cuda(), "index must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(index.dtype() == torch::kInt64, "index must be int64");
    TORCH_CHECK(dim == 1, "only dim=1 supported");

    // input must be contiguous (as given by embedding output)
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    // index can be strided (from expand), so allow non-contiguous but handle strides.
    // We need at least 3 dimensions.
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(index.dim() == 3, "index must be 3D");

    int64_t B = input.size(0);
    int64_t T = input.size(1);
    int64_t E = input.size(2);
    int64_t I = index.size(1);

    // Validate shapes: index size(0) must be B, index size(2) must be E
    TORCH_CHECK(index.size(0) == B, "index batch dim mismatch");
    TORCH_CHECK(index.size(2) == E, "index embedding dim mismatch");

    // Allocate output [B, I, E] (contiguous, like gather result)
    auto output = torch::empty({B, I, E}, input.options());

    // Get strides for input (contiguous)
    int64_t input_stride_b = input.stride(0);
    int64_t input_stride_t = input.stride(1);
    int64_t input_stride_e = input.stride(2);

    // Get strides for index (may be non-contiguous)
    int64_t index_stride_b = index.stride(0);
    int64_t index_stride_i = index.stride(1);
    int64_t index_stride_e = index.stride(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_gather_launcher(
        input.data_ptr<float>(),
        index.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        B, T, I, E,
        input_stride_b, input_stride_t, input_stride_e,
        index_stride_b, index_stride_i, index_stride_e,
        dim,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_embedding", &custom_embedding, "Custom embedding lookup",
          py::arg("input"), py::arg("weight"));
    m.def("custom_gather", &custom_gather, "Custom gather along dim=1",
          py::arg("input"), py::arg("index"), py::arg("dim") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);