#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_dim1_launcher(const float* input,
                                               float* output,
                                               int64_t N, int64_t C, int64_t H, int64_t W,
                                               int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                               cudaStream_t stream);

extern "C" void custom_sort_dim1_launcher(const float* input,
                                           float* sorted_values,
                                           int64_t* indices,
                                           int return_key,
                                           int64_t N, int64_t C, int64_t H, int64_t W,
                                           int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
                                           cudaStream_t stream);

extern "C" void custom_split_dim1_offsets(int64_t C, int64_t chunk_size,
                                         int64_t* offsets_out, int64_t* sizes_out,
                                         int* num_chunks_out);

// Helper to check common tensor requirements
static void check_cuda_float32_4d(const torch::Tensor& x, const char* op_name) {
    TORCH_CHECK(x.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, op_name, ": input must be float32");
    TORCH_CHECK(x.dim() == 4, op_name, ": input must be 4D (N,C,H,W)");
}

static torch::Tensor custom_softmax_dim1(torch::Tensor x) {
    check_cuda_float32_4d(x, "custom_softmax_dim1");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    return output;
}

static py::tuple custom_sort_dim1(torch::Tensor x, bool return_key = false) {
    check_cuda_float32_4d(x, "custom_sort_dim1");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto sorted_values = torch::empty_like(x);
    torch::Tensor indices;
    if (return_key) {
        indices = torch::empty_like(x, x.options().dtype(torch::kInt64));
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_sort_dim1_launcher(
        x.data_ptr<float>(),
        sorted_values.data_ptr<float>(),
        return_key ? indices.data_ptr<int64_t>() : nullptr,
        return_key ? 1 : 0,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream
    );

    if (return_key) {
        return py::make_tuple(sorted_values, indices);
    } else {
        return py::make_tuple(sorted_values);
    }
}

static py::tuple custom_split_dim1(torch::Tensor x, int64_t chunk_size) {
    check_cuda_float32_4d(x, "custom_split_dim1");
    TORCH_CHECK(chunk_size > 0, "custom_split_dim1: chunk_size must be positive");

    int64_t C = x.size(1);
    int num_chunks = (int)((C + chunk_size - 1) / chunk_size);

    std::vector<int64_t> offsets(num_chunks);
    std::vector<int64_t> sizes(num_chunks);
    int num_chunks_from_cuda = 0;

    // Call helper to compute offsets/sizes (host function, no CUDA needed)
    custom_split_dim1_offsets(C, chunk_size, offsets.data(), sizes.data(), &num_chunks_from_cuda);
    TORCH_CHECK(num_chunks_from_cuda == num_chunks, "custom_split_dim1: chunk calculation mismatch");

    py::tuple result(num_chunks);
    for (int i = 0; i < num_chunks; ++i) {
        // Use narrow to create a view along dim=1, sharing storage
        result[i] = x.narrow(1, offsets[i], sizes[i]);
    }
    return result;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax_dim1", &custom_softmax_dim1,
          "Custom softmax along dim=1 for 4D tensor",
          py::arg("x"));

    m.def("custom_sort_dim1", &custom_sort_dim1,
          "Custom sort along dim=1 for 4D tensor; returns tuple of (sorted_values, indices) if return_key=true else (sorted_values,)",
          py::arg("x"),
          py::arg("return_key") = false);

    m.def("custom_split_dim1", &custom_split_dim1,
          "Split tensor along dim=1 into views of given chunk_size",
          py::arg("x"),
          py::arg("chunk_size"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
