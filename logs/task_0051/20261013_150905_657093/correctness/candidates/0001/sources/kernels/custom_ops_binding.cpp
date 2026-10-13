#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 custom_ops.cu 导出的 launcher
extern "C" void embedding_forward_cuda(
    const int64_t* indices,
    const float* weight,
    float* output,
    int N,
    int D,
    int E,
    int num_embeddings,
    cudaStream_t stream);

extern "C" void reflection_pad2d_forward_cuda(
    const float* input,
    float* output,
    int N,
    int D,
    int W_in,
    int W_out,
    int pad_l,
    int pad_r,
    cudaStream_t stream);

extern "C" void sub_scalar_forward_cuda(
    const float* input,
    float* output,
    float scalar,
    int64_t total,
    cudaStream_t stream);

// embedding_forward 包装
torch::Tensor embedding_forward(torch::Tensor indices, torch::Tensor weight) {
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    auto N = indices.size(0);
    auto D = indices.size(1);
    auto E = weight.size(1);
    int num_embeddings = weight.size(0);

    auto output = torch::empty({N, D, E}, indices.options().dtype(torch::kFloat32));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    embedding_forward_cuda(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D, E, num_embeddings,
        stream);

    return output;
}

// reflection_pad2d_forward 包装
torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t pad_l, int64_t pad_r) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (N, C, W)");

    int N = input.size(0);
    int D = input.size(1);
    int W_in = input.size(2);
    int pad_l_int = static_cast<int>(pad_l);
    int pad_r_int = static_cast<int>(pad_r);
    int W_out = W_in + pad_l_int + pad_r_int;

    TORCH_CHECK(pad_l_int >= 0 && pad_r_int >= 0, "padding must be non-negative");

    auto output = torch::empty({N, D, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    reflection_pad2d_forward_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, D, W_in, W_out, pad_l_int, pad_r_int,
        stream);

    return output;
}

// sub_scalar_forward 包装
torch::Tensor sub_scalar_forward(torch::Tensor input, double scalar) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int64_t total = input.numel();
    float scalar_f = static_cast<float>(scalar);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sub_scalar_forward_cuda(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        scalar_f,
        total,
        stream);

    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward",
          py::arg("indices"), py::arg("weight"));
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward, "ReflectionPad2d forward",
          py::arg("input"), py::arg("pad_l"), py::arg("pad_r"));
    m.def("sub_scalar_forward", &sub_scalar_forward, "Sub scalar forward",
          py::arg("input"), py::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
