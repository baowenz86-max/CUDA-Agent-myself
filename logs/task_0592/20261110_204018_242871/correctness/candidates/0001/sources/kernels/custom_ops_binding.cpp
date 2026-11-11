#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" int cat_forward_launcher(const float* x1,
                                    const float* x2,
                                    float* output,
                                    long long x1_elements,
                                    long long x2_elements,
                                    int dim,
                                    cudaStream_t stream);

extern "C" int ceil_forward_launcher(const float* input,
                                     float* output,
                                     long long n,
                                     cudaStream_t stream);

// Tensor wrapper for cat_forward
torch::Tensor cat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    // Validate inputs
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(dim == 0, "Only dim=0 is supported");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "x1 and x2 must have the same shape");

    // Ensure contiguous (launcher assumes memory-contiguous layout for dim=0)
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();

    // Compute output shape: [x1.size(0)+x2.size(0), x1.size(1), ...]
    auto out_sizes = x1_contig.sizes().vec();
    out_sizes[0] = x1_contig.size(0) + x2_contig.size(0);
    auto output = torch::empty(out_sizes, x1_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    long long x1_elements = x1_contig.numel();
    long long x2_elements = x2_contig.numel();
    int status = cat_forward_launcher(
        x1_contig.data_ptr<float>(),
        x2_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        x1_elements,
        x2_elements,
        static_cast<int>(dim),
        stream
    );

    TORCH_CHECK(status == 0, "cat_forward kernel launch failed with status ", status);
    return output;
}

// Tensor wrapper for ceil_forward
torch::Tensor ceil_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int status = ceil_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x.numel()),
        stream
    );

    TORCH_CHECK(status == 0, "ceil_forward kernel launch failed with status ", status);
    return output;
}

// Registration functions
void register_cat_forward(pybind11::module& m) {
    m.def("cat_forward", &cat_forward,
          "Concatenate two CUDA float32 tensors along dim=0",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
}

void register_ceil_forward(pybind11::module& m) {
    m.def("ceil_forward", &ceil_forward,
          "Element-wise ceil for CUDA float32 tensor",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(cat_forward, register_cat_forward);
REGISTER_BINDING(ceil_forward, register_ceil_forward);
