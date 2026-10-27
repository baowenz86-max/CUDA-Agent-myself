#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(
    float* output,
    const float* input,
    float min_val,
    float max_val,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t numel,
    cudaStream_t stream);

extern "C" void add_scalar_forward_launcher(
    float* output,
    const float* input,
    float scalar,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3,
    int64_t stride0, int64_t stride1, int64_t stride2, int64_t stride3,
    int64_t numel,
    cudaStream_t stream);

// Helper function to extract strides and dimensions from a 4D tensor
static void get_tensor_geometry(const torch::Tensor& t, int64_t& dim0, int64_t& dim1, int64_t& dim2, int64_t& dim3,
                                int64_t& stride0, int64_t& stride1, int64_t& stride2, int64_t& stride3) {
    TORCH_CHECK(t.dim() == 4, "Expected 4D tensor, got ", t.dim(), "D");
    dim0 = t.size(0);
    dim1 = t.size(1);
    dim2 = t.size(2);
    dim3 = t.size(3);
    stride0 = t.stride(0);
    stride1 = t.stride(1);
    stride2 = t.stride(2);
    stride3 = t.stride(3);
}

// Wrapper for clamp: output = clamp(x, min, max)
torch::Tensor clamp_forward(const torch::Tensor& x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D tensor");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t dim0, dim1, dim2, dim3;
    int64_t stride0, stride1, stride2, stride3;
    get_tensor_geometry(x, dim0, dim1, dim2, dim3, stride0, stride1, stride2, stride3);
    int64_t numel = x.numel();

    clamp_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        numel,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clamp_forward_launcher failed: ", cudaGetErrorString(err));
    return output;
}

// Wrapper for add_scalar: output = x + scalar
torch::Tensor add_scalar_forward(const torch::Tensor& x, double scalar) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D tensor");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t dim0, dim1, dim2, dim3;
    int64_t stride0, stride1, stride2, stride3;
    get_tensor_geometry(x, dim0, dim1, dim2, dim3, stride0, stride1, stride2, stride3);
    int64_t numel = x.numel();

    add_scalar_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<float>(scalar),
        dim0, dim1, dim2, dim3,
        stride0, stride1, stride2, stride3,
        numel,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "add_scalar_forward_launcher failed: ", cudaGetErrorString(err));
    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward,
          "Clamp tensor values elementwise",
          py::arg("x"), py::arg("min"), py::arg("max"));
    m.def("add_scalar_forward", &add_scalar_forward,
          "Add scalar to tensor elementwise",
          py::arg("x"), py::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
