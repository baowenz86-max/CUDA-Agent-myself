#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int height,
    int width,
    cudaStream_t stream
);

extern "C" void sort_descending_dim1_forward_launcher(
    const float* input,
    float* values_out,
    int64_t* indices_out,
    int batch,
    int height,
    int width,
    cudaStream_t stream
);

// Softmax wrapper
torch::Tensor softmax_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, height, width)");

    int batch = x.size(0);
    int height = x.size(1);
    int width = x.size(2);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim1_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        height,
        width,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return output;
}

// Sort wrapper: returns a tuple (values, indices)
std::tuple<torch::Tensor, torch::Tensor> sort_descending_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, height, width)");

    int batch = x.size(0);
    int height = x.size(1);
    int width = x.size(2);

    auto values = torch::empty_like(x);
    auto indices = torch::empty({batch, height, width}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sort_descending_dim1_forward_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        batch,
        height,
        width,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("softmax_dim1_forward", &softmax_dim1_forward, "Softmax along dim=1",
          py::arg("x"));
    m.def("sort_descending_dim1_forward", &sort_descending_dim1_forward, "Stable descending sort along dim=1",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
