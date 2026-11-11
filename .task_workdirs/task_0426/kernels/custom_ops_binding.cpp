#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_abs_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream
);

extern "C" void custom_flip_launcher(
    float* output,
    const float* input,
    int64_t dims,
    int64_t dim2,
    const int64_t* shape,
    const int64_t* strides,
    int64_t ndim,
    int64_t numel,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// custom_abs_forward: element-wise absolute value
torch::Tensor custom_abs_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(x);

    cudaStream_t stream = getCurrentStream();
    custom_abs_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_abs_forward launch failed: ", cudaGetErrorString(err));

    return output;
}

// custom_flip_forward: flip along specified dimensions
torch::Tensor custom_flip_forward(torch::Tensor x, int64_t dims, int64_t dim2) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dim() >= 1, "Input must have at least 1 dimension");

    int64_t ndim = x.dim();
    TORCH_CHECK(dims >= -ndim && dims < ndim, "dims out of range");
    TORCH_CHECK(dim2 >= -ndim && dim2 < ndim, "dim2 out of range");

    // Normalize negative dimensions
    if (dims < 0) dims += ndim;
    if (dim2 < 0) dim2 += ndim;

    // Prepare shape and strides on host
    std::vector<int64_t> shape_host(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> strides_host(x.strides().begin(), x.strides().end());

    // Allocate device memory for shape and strides
    int64_t* shape_dev = nullptr;
    int64_t* strides_dev = nullptr;
    cudaMalloc(&shape_dev, ndim * sizeof(int64_t));
    cudaMalloc(&strides_dev, ndim * sizeof(int64_t));
    TORCH_CHECK(shape_dev != nullptr && strides_dev != nullptr, "cudaMalloc failed");

    cudaStream_t stream = getCurrentStream();
    cudaMemcpyAsync(shape_dev, shape_host.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(strides_dev, strides_host.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice, stream);

    auto output = torch::empty_like(x);

    // Call launcher
    custom_flip_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        dims,
        dim2,
        shape_dev,
        strides_dev,
        ndim,
        x.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_flip_forward launch failed: ", cudaGetErrorString(err));

    // Free device memory (stream may still be using it, but cudaFree after kernel completes is okay if we synchronize later)
    // We need to ensure kernel finishes before freeing. cudaFree on device memory used by a kernel can be called after kernel completion.
    // Since we don't explicitly synchronize, we can rely on PyTorch's stream synchronization later, but to be safe, we can use cudaStreamSynchronize or just keep memory until after kernel completes.
    // Simpler: we can use cudaFree after the kernel potentially uses the memory, but it's not safe without synchronization.
    // Use cudaStreamSynchronize(stream) before freeing. This adds overhead but is correct.
    cudaStreamSynchronize(stream);
    cudaFree(shape_dev);
    cudaFree(strides_dev);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("custom_abs_forward", &custom_abs_forward, "Element-wise absolute value",
          py::arg("x"));
    m.def("custom_flip_forward", &custom_flip_forward, "Flip along specified dimensions",
          py::arg("x"), py::arg("dims"), py::arg("dim2") = 3);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
