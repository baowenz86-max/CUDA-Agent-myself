#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void concat_dim_launcher(
    float* out,
    const float* x1,
    const float* x2,
    long long total_elems,
    int dim,
    long long out_s0, long long out_s1, long long out_s2, long long out_s3,
    long long x1_s0, long long x1_s1, long long x1_s2, long long x1_s3,
    long long x2_s0, long long x2_s1, long long x2_s2, long long x2_s3,
    long long c1,
    cudaStream_t stream
);

torch::Tensor concat_dim(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == 4, "concat_dim only supports 4D tensors");
    TORCH_CHECK(x2.dim() == 4, "concat_dim only supports 4D tensors");
    TORCH_CHECK(dim >= 0 && dim < 4, "dim out of range");
    for (int i = 0; i < 4; ++i) {
        if (i != dim) {
            TORCH_CHECK(x1.size(i) == x2.size(i), "x1 and x2 must have same size except along dim");
        }
    }

    // Output shape doubles along dim
    std::vector<int64_t> out_sizes(4);
    for (int i = 0; i < 4; ++i) {
        out_sizes[i] = (i == dim) ? x1.size(i) + x2.size(i) : x1.size(i);
    }
    auto out = torch::empty(out_sizes, x1.options()); // contiguous

    // Get sizes and strides as long long scalars
    long long out_s0 = out.stride(0);
    long long out_s1 = out.stride(1);
    long long out_s2 = out.stride(2);
    long long out_s3 = out.stride(3);

    long long x1_s0 = x1.stride(0);
    long long x1_s1 = x1.stride(1);
    long long x1_s2 = x1.stride(2);
    long long x1_s3 = x1.stride(3);

    long long x2_s0 = x2.stride(0);
    long long x2_s1 = x2.stride(1);
    long long x2_s2 = x2.stride(2);
    long long x2_s3 = x2.stride(3);

    long long c1 = x1.size(dim); // size of x1 along dim

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    concat_dim_launcher(
        out.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.numel(),
        static_cast<int>(dim),
        out_s0, out_s1, out_s2, out_s3,
        x1_s0, x1_s1, x1_s2, x1_s3,
        x2_s0, x2_s1, x2_s2, x2_s3,
        c1,
        stream
    );

    return out;
}

torch::Tensor rot90_k1_dims23(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "rot90_k1_dims23 requires 4D tensor");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto s0 = x.stride(0);
    auto s1 = x.stride(1);
    auto s2 = x.stride(2);
    auto s3 = x.stride(3);

    std::vector<int64_t> sizes = {B, C, W, H};
    std::vector<int64_t> strides = {s0, s1, s3, s2};

    return x.as_strided(sizes, strides);
}

void register_custom_ops(pybind11::module& m) {
    m.def("concat_dim", &concat_dim, "Concatenate two tensors along a dimension",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("rot90_k1_dims23", &rot90_k1_dims23, "Rotate tensor 90 degrees in dims 2 and 3 (view)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
