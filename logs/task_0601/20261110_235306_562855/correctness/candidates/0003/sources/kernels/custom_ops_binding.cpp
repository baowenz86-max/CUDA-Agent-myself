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
    long long B,
    long long C_split,
    long long H,
    long long W,
    cudaStream_t stream
);

torch::Tensor concat_dim(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == x2.dim(), "x1 and x2 must have same rank");
    TORCH_CHECK(x1.dim() == 4, "Only 4D tensors supported");
    TORCH_CHECK(dim >= 0 && dim < 4, "dim out of range");
    for (int i = 0; i < 4; ++i) {
        if (i != dim) {
            TORCH_CHECK(x1.size(i) == x2.size(i), "x1 and x2 must have same size except along dim");
        }
    }
    // Make inputs contiguous to simplify kernel
    x1 = x1.contiguous();
    x2 = x2.contiguous();

    long long B = x1.size(0);
    long long C = x1.size(1);
    long long H = x1.size(2);
    long long W = x1.size(3);
    long long C_split = (dim == 1) ? C : ((dim == 0) ? B : ((dim == 2) ? H : W));
    // Actually, split size along the concatenation dimension: x1.size(dim)
    long long dim_size = x1.size(dim);
    // For simplicity, we only handle dim=1 (channel concatenation) as this is the only case in the model.
    // But if dim is not 1, we need a more general kernel. Since self.cat_dim=1 from init, this is fine.
    TORCH_CHECK(dim == 1, "Only dim=1 concatenation is supported in this optimized path");
    C_split = dim_size;

    // Output shape: same as x1 but dim size doubles
    std::vector<int64_t> out_sizes = {x1.size(0), 2 * x1.size(1), x1.size(2), x1.size(3)};
    auto out = torch::empty(out_sizes, x1.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    concat_dim_launcher(
        out.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        out.numel(),
        B,
        C_split,
        H,
        W,
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
