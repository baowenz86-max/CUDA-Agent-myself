#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reduce_max_4d_launcher(
    const float* input,
    float* output,
    int64_t dim,
    const int64_t* x_dims,
    const int64_t* x_strides,
    cudaStream_t stream
);

extern "C" void reduce_prod_4d_launcher(
    const float* input,
    float* output,
    int64_t dim,
    const int64_t* x_dims,
    const int64_t* x_strides,
    cudaStream_t stream
);

torch::Tensor reduce_max_kernel_forward(torch::Tensor x, int64_t dim) {
    // 输入检查
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional");

    // 处理负 dim
    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    // 准备维度数组和步长数组（CPU 上的小数组）
    int64_t x_dims[4];
    int64_t x_strides[4];
    for (int i = 0; i < 4; ++i) {
        x_dims[i] = x.size(i);
        x_strides[i] = x.stride(i);
    }

    // 构建输出形状：dim 维为1，其余不变
    std::vector<int64_t> output_shape;
    output_shape.reserve(4);
    for (int i = 0; i < 4; ++i) {
        if (i == dim) {
            output_shape.push_back(1);
        } else {
            output_shape.push_back(x.size(i));
        }
    }

    // 分配输出张量
    auto output = torch::empty(output_shape, x.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    reduce_max_4d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        x_dims,
        x_strides,
        stream
    );

    // 检查 CUDA 错误
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return output;
}

torch::Tensor reduce_prod_kernel_forward(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4-dimensional");

    if (dim < 0) {
        dim += x.dim();
    }
    TORCH_CHECK(dim >= 0 && dim < x.dim(), "dim out of range");

    int64_t x_dims[4];
    int64_t x_strides[4];
    for (int i = 0; i < 4; ++i) {
        x_dims[i] = x.size(i);
        x_strides[i] = x.stride(i);
    }

    std::vector<int64_t> output_shape;
    output_shape.reserve(4);
    for (int i = 0; i < 4; ++i) {
        if (i == dim) {
            output_shape.push_back(1);
        } else {
            output_shape.push_back(x.size(i));
        }
    }

    auto output = torch::empty(output_shape, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reduce_prod_4d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        dim,
        x_dims,
        x_strides,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return output;
}

void register_reduce_ops(pybind11::module& m) {
    m.def("reduce_max_kernel_forward", &reduce_max_kernel_forward,
          "Max reduction along dim with keepdim=True",
          py::arg("x"), py::arg("dim"));
    m.def("reduce_prod_kernel_forward", &reduce_prod_kernel_forward,
          "Product reduction along dim with keepdim=True",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(reduce_ops, register_reduce_ops);
