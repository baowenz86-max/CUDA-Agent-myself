#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu6_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int ndim,
    const int64_t* dims,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

extern "C" void sum_dim1_forward_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

extern "C" void expand_forward_launcher(
    const float* input,
    float* output,
    int64_t B,
    int64_t out_features,
    int64_t H,
    int64_t W,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

// Helper to copy a host vector to device and return device pointer
static int64_t* copy_vec_to_device(const std::vector<int64_t>& vec, cudaStream_t stream) {
    int64_t* d_ptr = nullptr;
    cudaError_t err = cudaMalloc(&d_ptr, vec.size() * sizeof(int64_t));
    TORCH_CHECK(err == cudaSuccess, "cudaMalloc failed: ", cudaGetErrorString(err));
    err = cudaMemcpyAsync(d_ptr, vec.data(), vec.size() * sizeof(int64_t), cudaMemcpyHostToDevice, stream);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyAsync failed: ", cudaGetErrorString(err));
    return d_ptr;
}

// Helper to free device memory after kernel launch
static void free_device_ptr(int64_t* d_ptr) {
    if (d_ptr) {
        cudaError_t err = cudaFree(d_ptr);
        TORCH_CHECK(err == cudaSuccess, "cudaFree failed: ", cudaGetErrorString(err));
    }
}

torch::Tensor relu6_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu6_forward: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu6_forward: input must be float32");
    TORCH_CHECK(x.dim() >= 1, "relu6_forward: input must have at least 1 dimension");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t total_elements = x.numel();
    int ndim = x.dim();
    std::vector<int64_t> dims(x.sizes().begin(), x.sizes().end());
    std::vector<int64_t> input_strides(x.strides().begin(), x.strides().end());
    std::vector<int64_t> output_strides(output.strides().begin(), output.strides().end());

    // Allocate device memory for dims and strides
    int64_t* d_dims = copy_vec_to_device(dims, stream);
    int64_t* d_input_strides = copy_vec_to_device(input_strides, stream);
    int64_t* d_output_strides = copy_vec_to_device(output_strides, stream);

    relu6_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        ndim,
        d_dims,
        d_input_strides,
        d_output_strides,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "relu6_forward kernel launch failed: ", cudaGetErrorString(err));

    // Free device memory after kernel (synchronous free; kernel may still be running,
    // but since we are not using the pointers after this, it is safe as long as
    // cudaFree is called after a synchronize or if we don't need the data.
    // To be safe, we synchronize before freeing. Performance impact is acceptable for correctness.
    cudaStreamSynchronize(stream);
    free_device_ptr(d_dims);
    free_device_ptr(d_input_strides);
    free_device_ptr(d_output_strides);

    return output;
}

torch::Tensor sum_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sum_dim1_forward: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "sum_dim1_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_dim1_forward: input must be 4D [B, C, H, W]");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty({B, 1, H, W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    std::vector<int64_t> input_strides(x.strides().begin(), x.strides().end());
    std::vector<int64_t> output_strides(output.strides().begin(), output.strides().end());

    int64_t* d_input_strides = copy_vec_to_device(input_strides, stream);
    int64_t* d_output_strides = copy_vec_to_device(output_strides, stream);

    sum_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        C,
        H,
        W,
        d_input_strides,
        d_output_strides,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sum_dim1_forward kernel launch failed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    free_device_ptr(d_input_strides);
    free_device_ptr(d_output_strides);

    return output;
}

torch::Tensor expand_forward(torch::Tensor x, int64_t out_features) {
    TORCH_CHECK(x.is_cuda(), "expand_forward: input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "expand_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "expand_forward: input must be 4D [B, 1, H, W]");
    TORCH_CHECK(x.size(1) == 1, "expand_forward: dimension 1 must be size 1");

    int64_t B = x.size(0);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    std::vector<int64_t> output_shape = {B, out_features, H, W};
    std::vector<int64_t> output_strides = {x.stride(0), 0, x.stride(2), x.stride(3)};
    auto output = torch::empty_strided(output_shape, output_strides, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    std::vector<int64_t> input_strides(x.strides().begin(), x.strides().end());

    int64_t* d_input_strides = copy_vec_to_device(input_strides, stream);
    int64_t* d_output_strides = copy_vec_to_device(output_strides, stream);

    expand_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B,
        out_features,
        H,
        W,
        d_input_strides,
        d_output_strides,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "expand_forward kernel launch failed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    free_device_ptr(d_input_strides);
    free_device_ptr(d_output_strides);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu6_forward", &relu6_forward, "ReLU6 activation", py::arg("x"));
    m.def("sum_dim1_forward", &sum_dim1_forward, "Sum over dimension 1", py::arg("x"));
    m.def("expand_forward", &expand_forward, "Expand dimension 1", py::arg("x"), py::arg("out_features"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
