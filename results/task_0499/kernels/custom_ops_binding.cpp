#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    const int64_t* sizes,
    const int64_t* strides,
    int ndim,
    int64_t numel,
    float min_val,
    float max_val,
    cudaStream_t stream
);

extern "C" void argmin_forward_launcher(
    const float* input,
    int64_t* output,
    const int64_t* input_sizes,
    const int64_t* input_strides,
    const int64_t* output_sizes,
    const int64_t* output_strides,
    int ndim,
    int dim,
    int64_t output_numel,
    cudaStream_t stream
);

// Helper to convert tensor sizes/strides to int64_t vectors
static std::vector<int64_t> get_int64_vector(torch::IntArrayRef arr) {
    return std::vector<int64_t>(arr.begin(), arr.end());
}

// Helper to allocate device memory and copy host data
static int64_t* copy_to_device(const std::vector<int64_t>& host_data, cudaStream_t stream) {
    if (host_data.empty()) return nullptr;
    size_t bytes = host_data.size() * sizeof(int64_t);
    int64_t* device_ptr = nullptr;
    cudaError_t err = cudaMalloc(&device_ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed: " + std::string(cudaGetErrorString(err)));
    }
    err = cudaMemcpyAsync(device_ptr, host_data.data(), bytes, cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        cudaFree(device_ptr);
        throw std::runtime_error("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(err)));
    }
    return device_ptr;
}

// clamp_forward implementation
torch::Tensor clamp_forward(torch::Tensor x, double min, double max) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    auto output = torch::empty_like(x);

    int ndim = static_cast<int>(x.dim());
    int64_t numel = x.numel();
    if (numel == 0) return output;

    auto sizes_vec = get_int64_vector(x.sizes());
    auto strides_vec = get_int64_vector(x.strides());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Copy sizes and strides to device
    int64_t* d_sizes = copy_to_device(sizes_vec, stream);
    int64_t* d_strides = copy_to_device(strides_vec, stream);

    clamp_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        d_sizes,
        d_strides,
        ndim,
        numel,
        static_cast<float>(min),
        static_cast<float>(max),
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_sizes);
        cudaFree(d_strides);
        throw std::runtime_error("clamp kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Note: d_sizes and d_strides are still needed by the async kernel, cannot free immediately.
    // Use cudaFree after stream synchronization? For simplicity, we can add a callback.
    // But for correctness, we need to keep them alive until kernel completes.
    // Since we cannot easily free after async, we can leak or use cudaStreamAddCallback.
    // Here we use cudaFree after cudaStreamSynchronize? That would be slow.
    // Alternatively, we can allocate with cudaMallocManaged? No, simpler: use cudaFree after a small sync?
    // Better: use cudaStreamAddCallback to free after completion.
    // For now, we will not free to avoid complexity; memory leak is acceptable for this task? But may cause failure.
    // We'll implement proper freeing using cudaStreamAddCallback.
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        int64_t* ptrs = static_cast<int64_t*>(userData);
        cudaFree(ptrs);
    }, d_sizes, 0);
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        int64_t* ptrs = static_cast<int64_t*>(userData);
        cudaFree(ptrs);
    }, d_strides, 0);

    return output;
}

// argmin_forward implementation
torch::Tensor argmin_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    int ndim = static_cast<int>(x.dim());
    TORCH_CHECK(ndim > 0, "argmin requires at least 1 dimension");

    if (dim < 0) {
        dim += ndim;
    }
    TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension out of range");

    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            output_shape.push_back(x.size(i));
        }
    }

    auto output = torch::empty(output_shape, x.options().dtype(torch::kInt64));

    int64_t output_numel = output.numel();
    if (output_numel == 0) return output;

    auto input_sizes = get_int64_vector(x.sizes());
    auto input_strides = get_int64_vector(x.strides());

    auto output_sizes = get_int64_vector(output.sizes());
    std::vector<int64_t> output_strides(output_sizes.size());
    if (!output_sizes.empty()) {
        output_strides.back() = 1;
        for (int i = static_cast<int>(output_sizes.size()) - 2; i >= 0; --i) {
            output_strides[i] = output_strides[i + 1] * output_sizes[i + 1];
        }
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Copy all arrays to device
    int64_t* d_input_sizes = copy_to_device(input_sizes, stream);
    int64_t* d_input_strides = copy_to_device(input_strides, stream);
    int64_t* d_output_sizes = copy_to_device(output_sizes, stream);
    int64_t* d_output_strides = copy_to_device(output_strides, stream);

    argmin_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<int64_t>(),
        d_input_sizes,
        d_input_strides,
        d_output_sizes,
        d_output_strides,
        ndim,
        static_cast<int>(dim),
        output_numel,
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input_sizes);
        cudaFree(d_input_strides);
        cudaFree(d_output_sizes);
        cudaFree(d_output_strides);
        throw std::runtime_error("argmin kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Free device memory after kernel completion using callbacks
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cudaFree(userData);
    }, d_input_sizes, 0);
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cudaFree(userData);
    }, d_input_strides, 0);
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cudaFree(userData);
    }, d_output_sizes, 0);
    cudaStreamAddCallback(stream, [](cudaStream_t stream, cudaError_t status, void* userData) {
        cudaFree(userData);
    }, d_output_strides, 0);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward,
          py::arg("x"), py::arg("min"), py::arg("max"));
    m.def("argmin_forward", &argmin_forward,
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
