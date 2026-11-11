#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    float* output,
    const float* input,
    int64_t num_elements,
    int ndims,
    const int64_t* sizes,
    const int64_t* in_strides,
    const int64_t* out_strides,
    cudaStream_t stream
);

extern "C" void selu_forward_launcher(
    float* output,
    const float* input,
    int64_t num_elements,
    int ndims,
    const int64_t* sizes,
    const int64_t* in_strides,
    const int64_t* out_strides,
    cudaStream_t stream
);

namespace {

torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int ndims = x.dim();
    std::vector<int64_t> sizes(ndims);
    std::vector<int64_t> in_strides(ndims);
    std::vector<int64_t> out_strides(ndims);
    for (int i = 0; i < ndims; ++i) {
        sizes[i] = x.size(i);
        in_strides[i] = x.stride(i);
        out_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    relu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        ndims,
        sizes.data(),
        in_strides.data(),
        out_strides.data(),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "relu_forward kernel launch failed");
    return output;
}

torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input must not be empty");

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int ndims = x.dim();
    std::vector<int64_t> sizes(ndims);
    std::vector<int64_t> in_strides(ndims);
    std::vector<int64_t> out_strides(ndims);
    for (int i = 0; i < ndims; ++i) {
        sizes[i] = x.size(i);
        in_strides[i] = x.stride(i);
        out_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    selu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        ndims,
        sizes.data(),
        in_strides.data(),
        out_strides.data(),
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "selu_forward kernel launch failed");
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "Elementwise ReLU forward", pybind11::arg("x"));
    m.def("selu_forward", &selu_forward, "Elementwise SELU forward", pybind11::arg("x"));
}

}  // namespace

REGISTER_BINDING(custom_ops, register_custom_ops);
