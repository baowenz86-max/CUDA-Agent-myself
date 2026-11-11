#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// External launcher declarations from custom_ops.cu
extern "C" void custom_softplus_launcher(float* out, const float* in, long long numel, cudaStream_t stream);
extern "C" void custom_cat_launcher(float* out, const float* x1, const float* x2,
                                    int64_t N, int64_t C1, int64_t C2, int64_t H, int64_t W,
                                    int64_t dim, cudaStream_t stream);

// Helper to check CUDA errors after kernel launch
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, msg, ": CUDA error: ", cudaGetErrorString(err));
    }
}

// Softplus wrapper
torch::Tensor custom_softplus(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "custom_softplus: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_softplus: input must be float32");

    // Make contiguous if needed (launcher expects contiguous data)
    auto input_contig = input.contiguous();

    // Allocate output same shape as input
    auto output = torch::empty_like(input_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    custom_softplus_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        input_contig.numel(),
        stream
    );

    check_cuda_error("custom_softplus");

    // Return output (may be non-contiguous if original input was non-contiguous, but shape same)
    // Since output is created from contiguous input, it's contiguous; but to preserve strided behavior we could return as is.
    // PyTorch operations usually preserve stride; but for correctness of shape, we return output.
    return output;
}

// Cat wrapper
torch::Tensor custom_cat(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda(), "custom_cat: inputs must be CUDA tensors");
    TORCH_CHECK(x1.dtype() == torch::kFloat32 && x2.dtype() == torch::kFloat32, "custom_cat: inputs must be float32");
    TORCH_CHECK(x1.sizes() == x2.sizes(), "custom_cat: input shapes must match");
    TORCH_CHECK(dim == 1, "custom_cat: only dim=1 is supported for this task");

    // Make inputs contiguous
    auto x1_contig = x1.contiguous();
    auto x2_contig = x2.contiguous();

    // Get dimensions (4D tensors [N, C, H, W])
    int64_t N = x1_contig.size(0);
    int64_t C1 = x1_contig.size(1);
    int64_t C2 = x2_contig.size(1);
    int64_t H = x1_contig.size(2);
    int64_t W = x1_contig.size(3);

    // Output shape: [N, C1+C2, H, W]
    auto output = torch::empty({N, C1 + C2, H, W}, x1_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch cat kernel
    custom_cat_launcher(
        output.data_ptr<float>(),
        x1_contig.data_ptr<float>(),
        x2_contig.data_ptr<float>(),
        N, C1, C2, H, W,
        dim,
        stream
    );

    check_cuda_error("custom_cat");

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softplus", &custom_softplus, "Custom softplus forward", py::arg("input"));
    m.def("custom_cat", &custom_cat, "Custom concatenation along dim=1", py::arg("x1"), py::arg("x2"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
