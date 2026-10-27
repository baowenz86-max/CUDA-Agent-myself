#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void mean_dim_launcher(const float* input, float* output, int64_t dim,
                                 int64_t B, int64_t C, int64_t H, int64_t W,
                                 cudaStream_t stream);
extern "C" void gelu_launcher(const float* input, float* output, int64_t n,
                             cudaStream_t stream);

torch::Tensor mean_dim(torch::Tensor x, torch::Tensor dim) {
    TORCH_CHECK(x.is_cuda(), "mean_dim: x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "mean_dim: x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mean_dim: x must be float32");
    TORCH_CHECK(x.dim() == 4, "mean_dim: x must be 4D");

    TORCH_CHECK(dim.is_cuda(), "mean_dim: dim must be a CUDA tensor");
    TORCH_CHECK(dim.is_contiguous(), "mean_dim: dim must be contiguous");
    TORCH_CHECK(dim.dtype() == torch::kInt64, "mean_dim: dim must be int64");
    TORCH_CHECK(dim.numel() == 1, "mean_dim: dim must be a scalar tensor");

    int64_t dim_val;
    cudaMemcpy(&dim_val, dim.data_ptr<int64_t>(), sizeof(int64_t), cudaMemcpyDeviceToHost);
    TORCH_CHECK(dim_val >= -4 && dim_val < 4, "mean_dim: dim out of range");
    if (dim_val < 0) dim_val += 4;

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // compute output shape based on dim
    std::vector<int64_t> out_shape;
    if (dim_val == 0) {
        out_shape = {C, H, W};
    } else if (dim_val == 1) {
        out_shape = {B, H, W};
    } else if (dim_val == 2) {
        out_shape = {B, C, W};
    } else {
        out_shape = {B, C, H};
    }

    auto output = torch::empty(out_shape, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    mean_dim_launcher(x.data_ptr<float>(), output.data_ptr<float>(), dim_val,
                      B, C, H, W, stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor gelu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "gelu: x must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "gelu: x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "gelu: x must be float32");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    gelu_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "gelu kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("mean_dim", &mean_dim, "mean along a dimension",
          py::arg("x"), py::arg("dim"));
    m.def("gelu", &gelu, "GELU activation",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
