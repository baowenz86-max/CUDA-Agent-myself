// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// CUDA launchers exported from custom_ops.cu
extern "C" void equal_forward_launcher_cuda(const float* x, const float* y, bool* out, int64_t total_elements, cudaStream_t stream);
extern "C" void argsort_lastdim_forward_launcher_cuda(const float* x, int64_t* out, int64_t rows, int64_t cols, int64_t row_stride, cudaStream_t stream);

// Wrapper for equal_forward
torch::Tensor equal_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.sizes() == y.sizes(), "x and y must have the same shape");
    // Requirement: shapes are identical; no broadcasting needed.
    // Layout supports strides, but the launcher assumes contiguous per element ordering.
    // For correctness, ensure tensors are contiguous (or handle strides via contiguous call if needed).
    // Since input strides are contiguous in contract, but we do a defensive contiguous() to simplify launcher.
    auto x_contig = x.contiguous();
    auto y_contig = y.contiguous();
    auto output = torch::empty_like(x_contig, torch::TensorOptions().dtype(torch::kBool).device(x_contig.device()));
    
    int64_t total_elements = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    equal_forward_launcher_cuda(
        x_contig.data_ptr<float>(),
        y_contig.data_ptr<float>(),
        output.data_ptr<bool>(),
        total_elements,
        stream
    );
    return output;
}

// Wrapper for argsort_lastdim_forward
torch::Tensor argsort_lastdim_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() >= 1, "x must have at least one dimension");
    
    // Make contiguous to ensure row_stride == cols (launcher expects contiguous rows)
    auto x_contig = x.contiguous();
    int64_t rows = 1;
    for (int64_t i = 0; i < x_contig.dim() - 1; ++i) {
        rows *= x_contig.size(i);
    }
    int64_t cols = x_contig.size(-1);
    int64_t row_stride = cols; // contiguous after .contiguous()
    
    auto output = torch::empty_like(x_contig, torch::TensorOptions().dtype(torch::kInt64).device(x_contig.device()));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    argsort_lastdim_forward_launcher_cuda(
        x_contig.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        rows,
        cols,
        row_stride,
        stream
    );
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("equal_forward", &equal_forward, "Element-wise equality", py::arg("x"), py::arg("y"));
    m.def("argsort_lastdim_forward", &argsort_lastdim_forward, "Stable argsort along last dimension", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
