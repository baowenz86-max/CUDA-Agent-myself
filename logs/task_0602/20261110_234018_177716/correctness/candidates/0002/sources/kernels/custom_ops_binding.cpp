// PyTorch binding for softplus_forward and concat_forward
// All CUDA kernel implementations are in custom_ops.cu; this file only handles
// tensor validation, metadata marshalling, launcher invocation, and pybind registration.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void softplus_forward_launcher(
    float* output,
    const float* input,
    int64_t* output_strides,
    int64_t* input_strides,
    int64_t* shape,
    int ndims,
    int64_t numel,
    float beta,
    float threshold,
    cudaStream_t stream);

extern "C" void concat_forward_launcher(
    float* output,
    const float* input1,
    const float* input2,
    int64_t* out_shape,
    int64_t* in1_strides,
    int64_t* in2_strides,
    int ndims,
    int64_t concat_dim,
    int64_t input1_concat_size,
    int64_t input2_concat_size,
    cudaStream_t stream);

// Helper to allocate device buffer and copy int64 metadata
static int64_t* copy_metadata_to_device(const std::vector<int64_t>& metadata) {
    if (metadata.empty()) return nullptr;
    int64_t* d_ptr = nullptr;
    cudaMalloc(&d_ptr, metadata.size() * sizeof(int64_t));
    cudaMemcpy(d_ptr, metadata.data(), metadata.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
    return d_ptr;
}

// Wrapper for softplus_forward
torch::Tensor softplus_forward(torch::Tensor x, double beta, double threshold) {
    TORCH_CHECK(x.is_cuda(), "softplus_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "softplus_forward: input must be float32");
    TORCH_CHECK(x.dim() > 0, "softplus_forward: input must have at least 1 dimension");

    auto output = torch::empty_like(x);

    // Gather shape and strides
    int ndims = x.dim();
    std::vector<int64_t> shape(ndims);
    std::vector<int64_t> out_strides(ndims);
    std::vector<int64_t> in_strides(ndims);
    for (int i = 0; i < ndims; ++i) {
        shape[i] = x.size(i);
        out_strides[i] = output.stride(i);
        in_strides[i] = x.stride(i);
    }

    // Allocate device metadata
    int64_t* d_shape = copy_metadata_to_device(shape);
    int64_t* d_out_strides = copy_metadata_to_device(out_strides);
    int64_t* d_in_strides = copy_metadata_to_device(in_strides);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t numel = x.numel();
    softplus_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        d_out_strides,
        d_in_strides,
        d_shape,
        ndims,
        numel,
        static_cast<float>(beta),
        static_cast<float>(threshold),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softplus_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    // Free temporary device buffers
    cudaFree(d_shape);
    cudaFree(d_out_strides);
    cudaFree(d_in_strides);

    return output;
}

// Wrapper for concat_forward
torch::Tensor concat_forward(torch::Tensor x1, torch::Tensor x2, int64_t dim) {
    TORCH_CHECK(x1.is_cuda() && x2.is_cuda(), "concat_forward: inputs must be CUDA tensors");
    TORCH_CHECK(x1.dtype() == torch::kFloat32 && x2.dtype() == torch::kFloat32,
                "concat_forward: inputs must be float32");
    TORCH_CHECK(x1.dim() == x2.dim(), "concat_forward: inputs must have same number of dimensions");
    int ndims = x1.dim();
    TORCH_CHECK(ndims > 0, "concat_forward: inputs must have at least 1 dimension");

    // Normalize dim
    if (dim < 0) dim += ndims;
    TORCH_CHECK(dim >= 0 && dim < ndims, "concat_forward: dim out of range");

    // Check that non-concat dimensions match
    for (int i = 0; i < ndims; ++i) {
        if (i == dim) continue;
        TORCH_CHECK(x1.size(i) == x2.size(i), "concat_forward: sizes must match except in concat dim");
    }

    // Compute output shape
    std::vector<int64_t> out_shape(ndims);
    int64_t out_numel = 1;
    for (int i = 0; i < ndims; ++i) {
        if (i == dim) {
            out_shape[i] = x1.size(i) + x2.size(i);
        } else {
            out_shape[i] = x1.size(i);
        }
        out_numel *= out_shape[i];
    }

    // Create contiguous output
    auto output = torch::empty(out_shape, x1.options());

    // Gather input strides (output is contiguous, but launcher expects output shape only)
    std::vector<int64_t> in1_strides(ndims);
    std::vector<int64_t> in2_strides(ndims);
    for (int i = 0; i < ndims; ++i) {
        in1_strides[i] = x1.stride(i);
        in2_strides[i] = x2.stride(i);
    }

    // Allocate device metadata
    int64_t* d_out_shape = copy_metadata_to_device(out_shape);
    int64_t* d_in1_strides = copy_metadata_to_device(in1_strides);
    int64_t* d_in2_strides = copy_metadata_to_device(in2_strides);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t input1_concat_size = x1.size(dim);
    int64_t input2_concat_size = x2.size(dim);
    concat_forward_launcher(
        output.data_ptr<float>(),
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        d_out_shape,
        d_in1_strides,
        d_in2_strides,
        ndims,
        dim,
        input1_concat_size,
        input2_concat_size,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "concat_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    cudaFree(d_out_shape);
    cudaFree(d_in1_strides);
    cudaFree(d_in2_strides);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("softplus_forward", &softplus_forward,
          "Softplus forward (CUDA)",
          py::arg("x"),
          py::arg("beta") = 1.0,
          py::arg("threshold") = 20.0);
    m.def("concat_forward", &concat_forward,
          "Concatenate two tensors along dim (CUDA)",
          py::arg("x1"),
          py::arg("x2"),
          py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
