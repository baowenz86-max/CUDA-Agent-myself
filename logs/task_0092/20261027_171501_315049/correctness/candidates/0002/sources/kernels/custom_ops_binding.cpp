#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_forward_launcher(const float* x, float* out,
                                                int B, int C, int H, int W,
                                                int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                                cudaStream_t stream);

extern "C" void custom_sort_forward_launcher(const float* x, float* out_values, int64_t* out_indices,
                                             int B, int C, int H, int W,
                                             int64_t stride_b, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                             cudaStream_t stream);

extern "C" void custom_split_forward_launcher(const float* x, float* out_chunk,
                                              int B, int C, int H, int W,
                                              int64_t stride_b_in, int64_t stride_c_in, int64_t stride_h_in, int64_t stride_w_in,
                                              int offset, int chunk_size,
                                              int64_t stride_b_out, int64_t stride_c_out, int64_t stride_h_out, int64_t stride_w_out,
                                              cudaStream_t stream);

// Helper to check 4D float32 CUDA tensor
static void check_4d_float32_cuda(const torch::Tensor& x, const char* name) {
    TORCH_CHECK(x.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(x.dim() == 4, name, " must be 4D");
}

torch::Tensor custom_softmax_forward(torch::Tensor x, int64_t dim) {
    check_4d_float32_cuda(x, "x");
    TORCH_CHECK(dim == 1, "only dim=1 is supported for softmax");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_softmax_forward CUDA error: ", cudaGetErrorString(err));

    return out;
}

torch::return_types::sort custom_sort_forward(torch::Tensor x, int64_t dim) {
    check_4d_float32_cuda(x, "x");
    TORCH_CHECK(dim == 1, "only dim=1 is supported for sort");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto out_values = torch::empty_like(x);
    auto out_indices = torch::empty_like(x, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_sort_forward_launcher(
        x.data_ptr<float>(),
        out_values.data_ptr<float>(),
        out_indices.data_ptr<int64_t>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_sort_forward CUDA error: ", cudaGetErrorString(err));

    return {std::move(out_values), std::move(out_indices)};
}

std::vector<torch::Tensor> custom_split_forward(torch::Tensor x, int64_t split_size, int64_t dim) {
    check_4d_float32_cuda(x, "x");
    TORCH_CHECK(dim == 1, "only dim=1 is supported for split");
    TORCH_CHECK(split_size > 0, "split_size must be positive");

    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(C % split_size == 0, "C must be divisible by split_size");

    int num_chunks = (C + split_size - 1) / split_size;
    std::vector<torch::Tensor> chunks;
    chunks.reserve(num_chunks);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    for (int i = 0; i < num_chunks; ++i) {
        int offset = i * split_size;
        int chunk_size = std::min<int>(split_size, C - offset);
        auto out_chunk = torch::empty({B, chunk_size, H, W}, x.options());

        custom_split_forward_launcher(
            x.data_ptr<float>(),
            out_chunk.data_ptr<float>(),
            B, C, H, W,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            offset, chunk_size,
            out_chunk.stride(0), out_chunk.stride(1), out_chunk.stride(2), out_chunk.stride(3),
            stream
        );

        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "custom_split_forward CUDA error: ", cudaGetErrorString(err));

        chunks.push_back(std::move(out_chunk));
    }

    return chunks;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax_forward", &custom_softmax_forward,
          "Softmax over dim=1",
          py::arg("x"), py::arg("dim"));
    m.def("custom_sort_forward", &custom_sort_forward,
          "Sort over dim=1, returns (values, indices)",
          py::arg("x"), py::arg("dim"));
    m.def("custom_split_forward", &custom_split_forward,
          "Split tensor along dim=1 into chunks",
          py::arg("x"), py::arg("split_size"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
