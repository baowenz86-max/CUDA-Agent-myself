#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_launcher(const float* input,
                                     float* output,
                                     int64_t total_elements,
                                     cudaStream_t stream);

extern "C" void custom_cumsum_launcher(const float* input,
                                         float* output,
                                         int64_t dim0,
                                         int64_t inner_size,
                                         cudaStream_t stream);

// PyTorch wrapper for element-wise absolute value
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_abs: input must be float32");

    // Ensure contiguous for the kernel's flat indexing
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int64_t total_elements = x_contig.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_abs_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    // Return view with original strides if possible; otherwise return contiguous output
    if (x.is_contiguous()) {
        return output;
    } else {
        return output.as_strided(x.sizes(), x.strides());
    }
}

// PyTorch wrapper for cumulative sum along specified dimension
torch::Tensor custom_cumsum(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "custom_cumsum: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_cumsum: input must be float32");
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "custom_cumsum: dim out of range");

    // Only dim=0 is implemented in the kernel; contract expects dim=0
    TORCH_CHECK(dim == 0, "custom_cumsum: only dim=0 is supported by this kernel");

    // Ensure contiguous for the kernel's row-major assumption
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);

    int64_t dim0 = x_contig.size(0);
    int64_t inner_size = x_contig.numel() / dim0;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_cumsum_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        dim0,
        inner_size,
        stream
    );

    // Return view with original strides if possible
    if (x.is_contiguous()) {
        return output;
    } else {
        return output.as_strided(x.sizes(), x.strides());
    }
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs", &custom_abs, "Element-wise absolute value",
          py::arg("x"));
    m.def("custom_cumsum", &custom_cumsum, "Cumulative sum along dimension 0",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
