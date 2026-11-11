#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <vector>

// Declare launchers from custom_ops.cu (extern "C" to avoid name mangling)
extern "C" void cat_forward_launcher(
    const float* src1,
    const float* src2,
    float* dst,
    int dim,
    int ndims,
    const int* shape1,
    const int* shape2,
    const int* stride1,
    const int* stride2,
    const int* out_shape,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void prod_forward_launcher(
    const float* input,
    float* output,
    int dim,
    int ndims,
    const int* shape,
    const int* stride,
    const int* out_shape,
    int64_t out_numel,
    int reduce_size,
    cudaStream_t stream);

extern "C" void cat_prod_forward_launcher(
    const float* x1,
    const float* x2,
    float* output,
    int dim,
    int ndims,
    const int* shape1,
    const int* shape2,
    const int* stride1,
    const int* stride2,
    const int* out_shape,
    int64_t out_numel,
    int reduce_size1,
    int reduce_size2,
    cudaStream_t stream);

namespace {

// Helper to allocate a device buffer and copy host vector to it, returning device pointer
inline int* copy_int_vector_to_device(const std::vector<int>& vec, cudaStream_t stream) {
    if (vec.empty()) return nullptr;
    int* d_ptr = nullptr;
    cudaMalloc(&d_ptr, vec.size() * sizeof(int));
    cudaMemcpyAsync(d_ptr, vec.data(), vec.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    return d_ptr;
}

// Helper to free device memory after ensuring all previous work in stream is done
inline void free_device_ptr(int* ptr) {
    if (ptr) cudaFree(ptr);
}

} // anonymous namespace

// cat_forward: concatenate x1 and x2 along dimension dim
torch::Tensor cat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == x2.dim(), "x1 and x2 must have same number of dimensions");
    int ndims = x1.dim();
    TORCH_CHECK(dim >= -ndims && dim < ndims, "dim out of range");
    if (dim < 0) dim += ndims;
    int dim_i = static_cast<int>(dim);

    // Check that all non-concatenated dimensions match
    for (int d = 0; d < ndims; ++d) {
        if (d == dim_i) continue;
        TORCH_CHECK(x1.size(d) == x2.size(d), "size mismatch at non-concat dim", d);
    }

    // Compute output shape
    std::vector<int64_t> out_sizes(ndims);
    for (int d = 0; d < ndims; ++d) {
        if (d == dim_i) out_sizes[d] = x1.size(d) + x2.size(d);
        else out_sizes[d] = x1.size(d);
    }
    auto output = torch::empty(out_sizes, x1.options());

    // Gather shapes and strides as int vectors
    std::vector<int> shape1(ndims), shape2(ndims), stride1(ndims), stride2(ndims), out_shape(ndims);
    for (int d = 0; d < ndims; ++d) {
        shape1[d] = static_cast<int>(x1.size(d));
        shape2[d] = static_cast<int>(x2.size(d));
        stride1[d] = static_cast<int>(x1.stride(d));
        stride2[d] = static_cast<int>(x2.stride(d));
        out_shape[d] = static_cast<int>(out_sizes[d]);
    }
    int64_t total_elements = output.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Copy metadata to device
    int* d_shape1 = copy_int_vector_to_device(shape1, stream);
    int* d_shape2 = copy_int_vector_to_device(shape2, stream);
    int* d_stride1 = copy_int_vector_to_device(stride1, stream);
    int* d_stride2 = copy_int_vector_to_device(stride2, stream);
    int* d_out_shape = copy_int_vector_to_device(out_shape, stream);

    cat_forward_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_i,
        ndims,
        d_shape1,
        d_shape2,
        d_stride1,
        d_stride2,
        d_out_shape,
        total_elements,
        stream);

    // Free device memory after stream completion
    // Use cudaFree after synchronizing? Better to use cudaFreeAsync if available? cudaFree on a stream-ordered pointer is okay if we ensure no kernel uses it; but kernel may still be running. Use cudaFree after cudaStreamSynchronize for simplicity (performance not critical for metadata).
    // To avoid expensive synchronization, we rely on cudaFree being synchronous? It may block until kernel completion. We'll synchronize here and free.
    cudaStreamSynchronize(stream);
    free_device_ptr(d_shape1);
    free_device_ptr(d_shape2);
    free_device_ptr(d_stride1);
    free_device_ptr(d_stride2);
    free_device_ptr(d_out_shape);

    return output;
}

// prod_forward: product along dimension dim
torch::Tensor prod_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    int ndims = x.dim();
    TORCH_CHECK(dim >= -ndims && dim < ndims, "dim out of range");
    if (dim < 0) dim += ndims;
    int dim_i = static_cast<int>(dim);

    // Compute output shape: remove dim_i
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndims; ++d) {
        if (d == dim_i) continue;
        out_sizes.push_back(x.size(d));
    }
    auto output = torch::empty(out_sizes, x.options());

    // Gather shape and stride
    std::vector<int> shape(ndims), stride(ndims), out_shape(out_sizes.size());
    for (int d = 0; d < ndims; ++d) {
        shape[d] = static_cast<int>(x.size(d));
        stride[d] = static_cast<int>(x.stride(d));
    }
    for (size_t d = 0; d < out_sizes.size(); ++d) {
        out_shape[d] = static_cast<int>(out_sizes[d]);
    }
    int64_t out_numel = output.numel();
    int reduce_size = static_cast<int>(x.size(dim_i));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int* d_shape = copy_int_vector_to_device(shape, stream);
    int* d_stride = copy_int_vector_to_device(stride, stream);
    int* d_out_shape = copy_int_vector_to_device(out_shape, stream);

    prod_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_i,
        ndims,
        d_shape,
        d_stride,
        d_out_shape,
        out_numel,
        reduce_size,
        stream);

    cudaStreamSynchronize(stream);
    free_device_ptr(d_shape);
    free_device_ptr(d_stride);
    free_device_ptr(d_out_shape);

    return output;
}

// cat_prod_forward: fused concatenation and product
torch::Tensor cat_prod_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda(), "x1 must be a CUDA tensor");
    TORCH_CHECK(x2.is_cuda(), "x2 must be a CUDA tensor");
    TORCH_CHECK(x1.dtype() == torch::kFloat32, "x1 must be float32");
    TORCH_CHECK(x2.dtype() == torch::kFloat32, "x2 must be float32");
    TORCH_CHECK(x1.dim() == x2.dim(), "x1 and x2 must have same number of dimensions");
    int ndims = x1.dim();
    TORCH_CHECK(dim >= -ndims && dim < ndims, "dim out of range");
    if (dim < 0) dim += ndims;
    int dim_i = static_cast<int>(dim);

    for (int d = 0; d < ndims; ++d) {
        if (d == dim_i) continue;
        TORCH_CHECK(x1.size(d) == x2.size(d), "size mismatch at non-reduced dim", d);
    }

    // Output shape: remove dim_i
    std::vector<int64_t> out_sizes;
    for (int d = 0; d < ndims; ++d) {
        if (d == dim_i) continue;
        out_sizes.push_back(x1.size(d));
    }
    auto output = torch::empty(out_sizes, x1.options());

    std::vector<int> shape1(ndims), shape2(ndims), stride1(ndims), stride2(ndims), out_shape(out_sizes.size());
    for (int d = 0; d < ndims; ++d) {
        shape1[d] = static_cast<int>(x1.size(d));
        shape2[d] = static_cast<int>(x2.size(d));
        stride1[d] = static_cast<int>(x1.stride(d));
        stride2[d] = static_cast<int>(x2.stride(d));
    }
    for (size_t d = 0; d < out_sizes.size(); ++d) {
        out_shape[d] = static_cast<int>(out_sizes[d]);
    }
    int64_t out_numel = output.numel();
    int reduce_size1 = static_cast<int>(x1.size(dim_i));
    int reduce_size2 = static_cast<int>(x2.size(dim_i));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int* d_shape1 = copy_int_vector_to_device(shape1, stream);
    int* d_shape2 = copy_int_vector_to_device(shape2, stream);
    int* d_stride1 = copy_int_vector_to_device(stride1, stream);
    int* d_stride2 = copy_int_vector_to_device(stride2, stream);
    int* d_out_shape = copy_int_vector_to_device(out_shape, stream);

    cat_prod_forward_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        output.data_ptr<float>(),
        dim_i,
        ndims,
        d_shape1,
        d_shape2,
        d_stride1,
        d_stride2,
        d_out_shape,
        out_numel,
        reduce_size1,
        reduce_size2,
        stream);

    cudaStreamSynchronize(stream);
    free_device_ptr(d_shape1);
    free_device_ptr(d_shape2);
    free_device_ptr(d_stride1);
    free_device_ptr(d_stride2);
    free_device_ptr(d_out_shape);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("cat_forward", &cat_forward, "Concatenate two tensors along dim",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
    m.def("prod_forward", &prod_forward, "Product along dim",
          py::arg("x"), py::arg("dim"));
    m.def("cat_prod_forward", &cat_prod_forward, "Fused concatenation and product",
          py::arg("x1"), py::arg("x2"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
