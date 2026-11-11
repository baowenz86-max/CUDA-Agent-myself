#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_cat_launcher(float* out, const float* x1, const float* x2,
                                    int N, int C1, int C2, int H, int W,
                                    const int4 x1_strides, const int4 x2_strides,
                                    cudaStream_t stream);

extern "C" void custom_normalize_launcher(float* out, const float* x,
                                          int N, int C, int H, int W,
                                          const int4 strides,
                                          double p, double eps,
                                          cudaStream_t stream);

// Helper to pack strides into int4
template <typename T>
static int4 make_int4_from_strides(const T& strides) {
    // strides vector length must be 4
    TORCH_CHECK(strides.size() == 4, "Expected 4D strides");
    return make_int4(static_cast<int>(strides[0]),
                     static_cast<int>(strides[1]),
                     static_cast<int>(strides[2]),
                     static_cast<int>(strides[3]));
}

// custom_cat_forward: concatenates two 4D tensors along dim 1
torch::Tensor custom_cat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Basic checks
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == 4 && x2.dim() == 4, "Both inputs must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    // Shape checks: batch, height, width must match
    TORCH_CHECK(x1.size(0) == x2.size(0), "Batch dimension mismatch");
    TORCH_CHECK(x1.size(2) == x2.size(2), "Height dimension mismatch");
    TORCH_CHECK(x1.size(3) == x2.size(3), "Width dimension mismatch");

    // Make inputs contiguous if needed (launcher expects contiguous tensors with strides available;
    // but we pass strides, so it can work with non-contiguous as long as strides are valid.
    // To be safe and ensure correct indexing, we'll make contiguous copies if not contiguous.
    auto x1_contig = x1.is_contiguous() ? x1 : x1.contiguous();
    auto x2_contig = x2.is_contiguous() ? x2 : x2.contiguous();

    // Dimensions
    int64_t N = x1_contig.size(0);
    int64_t C1 = x1_contig.size(1);
    int64_t C2 = x2_contig.size(1);
    int64_t H = x1_contig.size(2);
    int64_t W = x1_contig.size(3);
    int64_t C = C1 + C2;

    // Allocate output as contiguous tensor
    auto out = torch::empty({N, C, H, W}, x1_contig.options());

    // Pack strides (in elements, not bytes)
    int4 x1_strides = make_int4_from_strides(x1_contig.strides());
    int4 x2_strides = make_int4_from_strides(x2_contig.strides());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_cat_launcher(
        out.data_ptr<float>(),
        x1_contig.data_ptr<float>(),
        x2_contig.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C1), static_cast<int>(C2),
        static_cast<int>(H), static_cast<int>(W),
        x1_strides, x2_strides,
        stream
    );

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_cat_launcher failed");

    return out;
}

// custom_normalize_forward: Lp normalization along dimension 1
torch::Tensor custom_normalize_forward(torch::Tensor x, double p, int64_t dim, double eps) {
    // Basic checks
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    // Make input contiguous if needed
    auto x_contig = x.is_contiguous() ? x : x.contiguous();

    // Dimensions
    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t H = x_contig.size(2);
    int64_t W = x_contig.size(3);

    // Allocate output with same shape and dtype
    auto out = torch::empty_like(x_contig);

    // Pack strides (contiguous input, strides in elements)
    int4 strides = make_int4_from_strides(x_contig.strides());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_normalize_launcher(
        out.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        strides,
        p, eps,
        stream
    );

    // Check for launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_normalize_launcher failed");

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_cat_forward", &custom_cat_forward,
          "Concatenates two 4D tensors along dimension 1",
          py::arg("x1"), py::arg("x2"), py::arg("dim") = 1);
    m.def("custom_normalize_forward", &custom_normalize_forward,
          "Lp normalization along dimension 1",
          py::arg("x"), py::arg("p") = 2.0, py::arg("dim") = 1, py::arg("eps") = 1e-12);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
