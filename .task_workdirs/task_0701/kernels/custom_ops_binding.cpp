#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void sum_launcher(const float* input,
                             const int64_t* dims,
                             const int64_t* strides,
                             int64_t ndim,
                             int64_t numel,
                             float* output,
                             cudaStream_t stream);

// PyTorch wrapper for split: returns tuple of tensors (views)
py::tuple split_forward(torch::Tensor x, int64_t split_size, int64_t axis) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(split_size > 0, "split_size must be positive");

    // Use torch.split to create views; this does not perform computation.
    auto splits = x.split(split_size, axis);

    // Convert std::vector<torch::Tensor> to py::tuple
    py::tuple result(splits.size());
    for (size_t i = 0; i < splits.size(); ++i) {
        result[i] = splits[i];
    }
    return result;
}

// PyTorch wrapper for sum: returns scalar tensor
torch::Tensor sum_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // Allocate output scalar tensor on same device
    auto output = torch::empty({}, x.options());

    // If tensor is empty, return zero
    if (x.numel() == 0) {
        output.zero_();
        return output;
    }

    // Gather dims and strides as int64_t arrays on CPU
    int64_t ndim = x.dim();
    std::vector<int64_t> dims(ndim);
    std::vector<int64_t> strides(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        dims[i] = x.size(i);
        strides[i] = x.stride(i);
    }

    // Copy dims and strides to GPU memory (launcher expects device pointers)
    // We can allocate temporary device buffers using torch::empty and copy from CPU,
    // or use cudaMalloc/cudaMemcpy. Since we must avoid torch ops for computation,
    // using cudaMalloc is fine (memory allocation is allowed).
    int64_t* d_dims = nullptr;
    int64_t* d_strides = nullptr;
    cudaMalloc(&d_dims, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_dims, dims.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, strides.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    sum_launcher(
        x.data_ptr<float>(),
        d_dims,
        d_strides,
        ndim,
        x.numel(),
        output.data_ptr<float>(),
        stream
    );

    // Free temporary device buffers (asynchronously, but launcher uses them before completion)
    cudaFree(d_dims);
    cudaFree(d_strides);

    // Synchronize? Not necessary; launcher copies output async within same stream.
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(py::module& m) {
    m.def("split", &split_forward, "Split tensor along axis (views)",
          py::arg("x"), py::arg("split_size"), py::arg("axis"));
    m.def("sum", &sum_forward, "Sum all elements",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
