#include <torch/types.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

namespace py = pybind11;

// Declare launchers from custom_ops.cu
extern "C" void custom_cat_launcher(
    float* output,
    const float* input1,
    const float* input2,
    int64_t dim,
    int64_t size1,
    int64_t size2,
    int64_t outer_size,
    int64_t inner_size,
    int64_t total_output_size,
    cudaStream_t stream
);

extern "C" void custom_nansum_launcher(
    float* output,
    const float* input,
    int64_t outer_size,
    int64_t reduction_size,
    int64_t inner_size,
    cudaStream_t stream
);

// custom_cat: concat two tensors along dim=1
torch::Tensor custom_cat(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Validate inputs
    TORCH_CHECK(x1.is_cuda(), "custom_cat: x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "custom_cat: x2 must be a CUDA tensor");
    TORCH_CHECK(x1.is_contiguous(), "custom_cat: x1 must be contiguous");
    TORCH_CHECK(x2.is_contiguous(), "custom_cat: x2 must be contiguous");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "custom_cat: x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "custom_cat: x2 must be float32");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "custom_cat: x1 and x2 must have the same shape");
    TORCH_CHECK(dim == 1, "custom_cat: only dim=1 is supported for this model");

    // Get shapes: x1 and x2 are [B, C, H, W] with B=32, C=3, H=1024, W=1024
    int64_t B = x1.size(0);
    int64_t C1 = x1.size(1);
    int64_t C2 = x2.size(1); // same as C1 since sizes equal
    int64_t H = x1.size(2);
    int64_t W = x1.size(3);

    // Prepare launcher parameters
    int64_t outer_size = B;
    int64_t size1 = C1;
    int64_t size2 = C2;
    int64_t inner_size = H * W;
    int64_t total_output_size = outer_size * (size1 + size2) * inner_size;

    // Allocate output tensor [B, C1+C2, H, W]
    auto output = torch::empty({B, size1 + size2, H, W}, x1.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_cat_launcher(
        output.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        dim,
        size1,
        size2,
        outer_size,
        inner_size,
        total_output_size,
        stream
    );

    return output;
}

// custom_nansum: sum along dim=1 ignoring NaNs
torch::Tensor custom_nansum(torch::Tensor x, int64_t dim) {
    // Validate input
    TORCH_CHECK(x.is_cuda(), "custom_nansum: x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_nansum: x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_nansum: x must be float32");
    TORCH_CHECK(dim == 1, "custom_nansum: only dim=1 is supported for this model");

    // Expected input shape [B, C, H, W] with C=6, B=32, H=1024, W=1024
    TORCH_CHECK(x.dim() == 4, "custom_nansum: expected 4D tensor");
    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    TORCH_CHECK(C == 6, "custom_nansum: expected reduction size 6");

    // Prepare launcher parameters
    int64_t outer_size = B;
    int64_t reduction_size = C;
    int64_t inner_size = H * W;

    // Allocate output tensor [B, H, W]
    auto output = torch::empty({B, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_nansum_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        outer_size,
        reduction_size,
        inner_size,
        stream
    );

    return output;
}

// Register both operators
void register_custom_ops(py::module& m) {
    m.def("custom_cat", &custom_cat,
          "Concatenate two tensors along dimension 1",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("custom_nansum", &custom_nansum,
          "Sum along dimension 1 ignoring NaNs",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
