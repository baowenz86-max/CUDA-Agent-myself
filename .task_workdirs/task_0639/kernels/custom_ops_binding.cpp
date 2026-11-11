#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void max_reduce_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int N, int C, int H, int W,
    const int64_t* x_strides,
    const int64_t* values_strides,
    const int64_t* indices_strides,
    cudaStream_t stream);

extern "C" void layer_norm_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    float eps,
    int N, int H, int D,
    const int64_t* input_strides,
    const int64_t* output_strides,
    cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor> max_reduce(torch::Tensor x, int64_t dim, bool keepdim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(keepdim, "Only keepdim=true is supported");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto values = torch::empty({N, 1, H, W}, x.options());
    auto indices = torch::empty({N, 1, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t x_strides[4] = {x.stride(0), x.stride(1), x.stride(2), x.stride(3)};
    int64_t values_strides[4] = {values.stride(0), values.stride(1), values.stride(2), values.stride(3)};
    int64_t indices_strides[4] = {indices.stride(0), indices.stride(1), indices.stride(2), indices.stride(3)};

    max_reduce_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N, C, H, W,
        x_strides,
        values_strides,
        indices_strides,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "max_reduce kernel launch failed: ", cudaGetErrorString(err));

    return {values, indices};
}

torch::Tensor layer_norm_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias, double eps) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N,1,H,W]");
    TORCH_CHECK(input.size(1) == 1, "Input dim 1 must be 1");
    TORCH_CHECK(weight.numel() == input.size(3), "Weight size must match last dim");
    TORCH_CHECK(bias.numel() == input.size(3), "Bias size must match last dim");

    int N = input.size(0);
    int H = input.size(2);
    int D = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t input_strides[4] = {input.stride(0), input.stride(1), input.stride(2), input.stride(3)};
    int64_t output_strides[4] = {output.stride(0), output.stride(1), output.stride(2), output.stride(3)};

    layer_norm_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(eps),
        N, H, D,
        input_strides,
        output_strides,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "layer_norm kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("max_reduce", &max_reduce, "Max reduce along dim=1 with keepdim=true",
          py::arg("x"), py::arg("dim"), py::arg("keepdim"));
    m.def("layer_norm_forward", &layer_norm_forward, "LayerNorm forward over last dim",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
