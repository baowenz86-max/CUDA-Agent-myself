#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void rnn_forward_launcher(
    const float* input,
    const float* weight_ih_l0,
    const float* weight_hh_l0,
    const float* bias_ih_l0,
    const float* bias_hh_l0,
    const float* weight_ih_l1,
    const float* weight_hh_l1,
    const float* bias_ih_l1,
    const float* bias_hh_l1,
    const float* hx,
    float* output,
    float* h_n,
    int seq_length,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor> rnn_forward(
    torch::Tensor input,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0,
    torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1,
    torch::Tensor bias_hh_l1,
    c10::optional<torch::Tensor> hx
) {
    // 参数检查
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    TORCH_CHECK(weight_ih_l0.is_cuda() && weight_ih_l0.is_contiguous() && weight_ih_l0.dtype() == torch::kFloat32, "weight_ih_l0 must be CUDA contiguous float32");
    TORCH_CHECK(weight_hh_l0.is_cuda() && weight_hh_l0.is_contiguous() && weight_hh_l0.dtype() == torch::kFloat32, "weight_hh_l0 must be CUDA contiguous float32");
    TORCH_CHECK(bias_ih_l0.is_cuda() && bias_ih_l0.is_contiguous() && bias_ih_l0.dtype() == torch::kFloat32, "bias_ih_l0 must be CUDA contiguous float32");
    TORCH_CHECK(bias_hh_l0.is_cuda() && bias_hh_l0.is_contiguous() && bias_hh_l0.dtype() == torch::kFloat32, "bias_hh_l0 must be CUDA contiguous float32");
    TORCH_CHECK(weight_ih_l1.is_cuda() && weight_ih_l1.is_contiguous() && weight_ih_l1.dtype() == torch::kFloat32, "weight_ih_l1 must be CUDA contiguous float32");
    TORCH_CHECK(weight_hh_l1.is_cuda() && weight_hh_l1.is_contiguous() && weight_hh_l1.dtype() == torch::kFloat32, "weight_hh_l1 must be CUDA contiguous float32");
    TORCH_CHECK(bias_ih_l1.is_cuda() && bias_ih_l1.is_contiguous() && bias_ih_l1.dtype() == torch::kFloat32, "bias_ih_l1 must be CUDA contiguous float32");
    TORCH_CHECK(bias_hh_l1.is_cuda() && bias_hh_l1.is_contiguous() && bias_hh_l1.dtype() == torch::kFloat32, "bias_hh_l1 must be CUDA contiguous float32");

    if (hx.has_value()) {
        TORCH_CHECK(hx.value().is_cuda(), "hx must be a CUDA tensor");
        TORCH_CHECK(hx.value().is_contiguous(), "hx must be contiguous");
        TORCH_CHECK(hx.value().dtype() == torch::kFloat32, "hx must be float32");
    }

    // 获取维度
    int seq_length = input.size(0);
    int batch_size = input.size(1);
    int input_size = input.size(2);
    int hidden_size = weight_ih_l0.size(0);
    int num_layers = 2; // 契约固定

    // 分配输出
    auto output = torch::empty({seq_length, batch_size, hidden_size}, input.options());
    auto h_n = torch::empty({num_layers, batch_size, hidden_size}, input.options());

    // 获取 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 调用 launcher
    const float* hx_ptr = hx.has_value() ? hx.value().data_ptr<float>() : nullptr;
    rnn_forward_launcher(
        input.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(),
        weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(),
        bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(),
        weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(),
        bias_hh_l1.data_ptr<float>(),
        hx_ptr,
        output.data_ptr<float>(),
        h_n.data_ptr<float>(),
        seq_length,
        batch_size,
        input_size,
        hidden_size,
        num_layers,
        stream
    );

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "rnn_forward_launcher failed with error: ", cudaGetErrorString(err));

    return std::make_tuple(output, h_n);
}

void register_custom_ops(pybind11::module& m) {
    m.def("rnn_forward", &rnn_forward,
          "RNN forward (custom CUDA implementation)",
          py::arg("input"),
          py::arg("weight_ih_l0"),
          py::arg("weight_hh_l0"),
          py::arg("bias_ih_l0"),
          py::arg("bias_hh_l0"),
          py::arg("weight_ih_l1"),
          py::arg("weight_hh_l1"),
          py::arg("bias_ih_l1"),
          py::arg("bias_hh_l1"),
          py::arg("hx") = py::none());
}

REGISTER_BINDING(custom_ops, register_custom_ops);
