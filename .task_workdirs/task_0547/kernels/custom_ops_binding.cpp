#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_triu_launcher(
    const float* x,
    float* out,
    int total_elements,
    int dim0,
    int dim1,
    long long int stride0,
    long long int stride1,
    long long int batch_stride,
    int diagonal,
    cudaStream_t stream
);

extern "C" void custom_abs_launcher(
    const float* x,
    float* out,
    int total_elements,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static inline cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// custom_triu forward
torch::Tensor custom_triu_forward(torch::Tensor x, int64_t diagonal = 1) {
    TORCH_CHECK(x.is_cuda(), "custom_triu: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_triu: input must be float32");
    TORCH_CHECK(x.dim() >= 2, "custom_triu: input must have at least 2 dimensions");

    // Shape parameters for last two dims
    int64_t height = x.size(-2);
    int64_t width = x.size(-1);
    TORCH_CHECK(height > 0 && width > 0, "custom_triu: invalid last dimensions");

    // Strides for last two dims (as long long to match launcher)
    long long int stride0 = static_cast<long long int>(x.stride(-2));
    long long int stride1 = static_cast<long long int>(x.stride(-1));

    // Batch stride: distance between consecutive batches (if any)
    // For non-batched case, batch_stride is 0 but unused.
    long long int batch_stride = 0;
    if (x.dim() > 2) {
        // stride of the dimension just before the last two
        batch_stride = static_cast<long long int>(x.stride(-3));
    }

    // Allocate output same shape, dtype, device
    auto out = torch::empty_like(x);

    int64_t total_elements = x.numel();
    TORCH_CHECK(total_elements <= INT_MAX, "custom_triu: tensor too large for int indexing");

    cudaStream_t stream = getCurrentCUDAStream();

    custom_triu_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(total_elements),
        static_cast<int>(height),
        static_cast<int>(width),
        stride0,
        stride1,
        batch_stride,
        static_cast<int>(diagonal),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_triu: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// custom_abs forward
torch::Tensor custom_abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_abs: input must be float32");

    auto out = torch::empty_like(x);

    int64_t total_elements = x.numel();
    TORCH_CHECK(total_elements <= INT_MAX, "custom_abs: tensor too large for int indexing");

    cudaStream_t stream = getCurrentCUDAStream();

    custom_abs_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(total_elements),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_triu", &custom_triu_forward,
          "Apply upper triangular mask with given diagonal",
          py::arg("x"), py::arg("diagonal") = 1);
    m.def("custom_abs", &custom_abs_forward,
          "Element-wise absolute value",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
