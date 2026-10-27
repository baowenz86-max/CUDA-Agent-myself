#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void group_norm_forward_cuda(
    const float* x, const float* weight, const float* bias,
    float* y, int N, int C, int L, int num_groups, float eps,
    cudaStream_t stream);

extern "C" void scalar_div_cuda(
    const float* x, float* y, int numel, float divisor, cudaStream_t stream);

extern "C" void lstm_forward_cuda(
    const float* x,
    const float* w_ih_l0, const float* w_hh_l0, const float* b_ih_l0, const float* b_hh_l0,
    const float* w_ih_l1, const float* w_hh_l1, const float* b_ih_l1, const float* b_hh_l1,
    const float* h0, const float* c0,
    float* output, float* h_n, float* c_n,
    int batch, int seq_len, int input_size, int hidden_size, int num_layers,
    cudaStream_t stream);

// group_norm_forward
static torch::Tensor group_norm_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
    int64_t num_groups, int64_t num_channels, double eps) {
    TORCH_CHECK(x.is_cuda(), "group_norm_forward: x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "group_norm_forward: weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "group_norm_forward: bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "group_norm_forward: x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "group_norm_forward: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "group_norm_forward: bias must be float32");

    TORCH_CHECK(weight.is_contiguous(), "group_norm_forward: weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "group_norm_forward: bias must be contiguous");
    auto x_contig = x.contiguous();
    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int L = x_contig.size(2);
    TORCH_CHECK(C == num_channels, "group_norm_forward: num_channels must match C dimension");
    TORCH_CHECK(weight.numel() == num_channels, "group_norm_forward: weight size must match num_channels");
    TORCH_CHECK(bias.numel() == num_channels, "group_norm_forward: bias size must match num_channels");

    auto y = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    group_norm_forward_cuda(
        x_contig.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(),
        y.data_ptr<float>(), N, C, L, static_cast<int>(num_groups), static_cast<float>(eps),
        stream);

    return y;
}

// permute_021: returns x.permute({0,2,1}) as a view, no data movement
static torch::Tensor permute_021(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "permute_021: x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "permute_021: x must be float32");
    TORCH_CHECK(x.dim() == 3, "permute_021: x must be 3D");
    return x.permute({0, 2, 1});
}

// lstm_forward
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> lstm_forward(
    torch::Tensor x,
    torch::Tensor weight_ih_l0, torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0, torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1, torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1, torch::Tensor bias_hh_l1,
    torch::Tensor h0, torch::Tensor c0,
    int64_t num_layers, int64_t hidden_size, bool batch_first) {
    TORCH_CHECK(x.is_cuda(), "lstm_forward: x must be CUDA");
    TORCH_CHECK(weight_ih_l0.is_cuda() && weight_hh_l0.is_cuda() &&
                bias_ih_l0.is_cuda() && bias_hh_l0.is_cuda() &&
                weight_ih_l1.is_cuda() && weight_hh_l1.is_cuda() &&
                bias_ih_l1.is_cuda() && bias_hh_l1.is_cuda() &&
                h0.is_cuda() && c0.is_cuda(),
                "lstm_forward: all tensors must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "lstm_forward: x must be float32");
    TORCH_CHECK(weight_ih_l0.dtype() == torch::kFloat32 &&
                weight_hh_l0.dtype() == torch::kFloat32 &&
                bias_ih_l0.dtype() == torch::kFloat32 &&
                bias_hh_l0.dtype() == torch::kFloat32 &&
                weight_ih_l1.dtype() == torch::kFloat32 &&
                weight_hh_l1.dtype() == torch::kFloat32 &&
                bias_ih_l1.dtype() == torch::kFloat32 &&
                bias_hh_l1.dtype() == torch::kFloat32 &&
                h0.dtype() == torch::kFloat32 && c0.dtype() == torch::kFloat32,
                "lstm_forward: all tensors must be float32");
    TORCH_CHECK(x.is_contiguous(), "lstm_forward: x must be contiguous");
    TORCH_CHECK(weight_ih_l0.is_contiguous() && weight_hh_l0.is_contiguous() &&
                bias_ih_l0.is_contiguous() && bias_hh_l0.is_contiguous() &&
                weight_ih_l1.is_contiguous() && weight_hh_l1.is_contiguous() &&
                bias_ih_l1.is_contiguous() && bias_hh_l1.is_contiguous() &&
                h0.is_contiguous() && c0.is_contiguous(),
                "lstm_forward: all tensors must be contiguous");

    TORCH_CHECK(x.dim() == 3, "lstm_forward: x must be 3D");
    TORCH_CHECK(batch_first, "lstm_forward: only batch_first=True supported");
    int batch = x.size(0);
    int seq_len = x.size(1);
    int input_size = x.size(2);
    int H = static_cast<int>(hidden_size);
    TORCH_CHECK(num_layers == 2, "lstm_forward: num_layers must be 2 (current implementation)");
    TORCH_CHECK(h0.size(0) == num_layers && h0.size(1) == batch && h0.size(2) == H,
                "lstm_forward: h0 shape mismatch");
    TORCH_CHECK(c0.size(0) == num_layers && c0.size(1) == batch && c0.size(2) == H,
                "lstm_forward: c0 shape mismatch");

    auto output = torch::empty({batch, seq_len, H}, x.options());
    auto h_n = torch::empty({num_layers, batch, H}, x.options());
    auto c_n = torch::empty({num_layers, batch, H}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    lstm_forward_cuda(
        x.data_ptr<float>(),
        weight_ih_l0.data_ptr<float>(), weight_hh_l0.data_ptr<float>(),
        bias_ih_l0.data_ptr<float>(), bias_hh_l0.data_ptr<float>(),
        weight_ih_l1.data_ptr<float>(), weight_hh_l1.data_ptr<float>(),
        bias_ih_l1.data_ptr<float>(), bias_hh_l1.data_ptr<float>(),
        h0.data_ptr<float>(), c0.data_ptr<float>(),
        output.data_ptr<float>(), h_n.data_ptr<float>(), c_n.data_ptr<float>(),
        batch, seq_len, input_size, H, static_cast<int>(num_layers),
        stream);

    return std::make_tuple(output, h_n, c_n);
}

// scalar_div: elementwise division by scalar
static torch::Tensor scalar_div(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "scalar_div: x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "scalar_div: x must be float32");
    auto x_contig = x.contiguous();
    int numel = x_contig.numel();
    auto y = torch::empty_like(x_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    scalar_div_cuda(x_contig.data_ptr<float>(), y.data_ptr<float>(), numel,
                    static_cast<float>(divisor), stream);
    return y;
}

// Registration
void register_custom_ops_binding(pybind11::module& m) {
    m.def("group_norm_forward", &group_norm_forward, "GroupNorm forward",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("num_groups"), pybind11::arg("num_channels"), pybind11::arg("eps"));
    m.def("permute_021", &permute_021, "Permute(0,2,1) view", pybind11::arg("x"));
    m.def("lstm_forward", &lstm_forward, "LSTM forward",
          pybind11::arg("x"),
          pybind11::arg("weight_ih_l0"), pybind11::arg("weight_hh_l0"),
          pybind11::arg("bias_ih_l0"), pybind11::arg("bias_hh_l0"),
          pybind11::arg("weight_ih_l1"), pybind11::arg("weight_hh_l1"),
          pybind11::arg("bias_ih_l1"), pybind11::arg("bias_hh_l1"),
          pybind11::arg("h0"), pybind11::arg("c0"),
          pybind11::arg("num_layers"), pybind11::arg("hidden_size"), pybind11::arg("batch_first"));
    m.def("scalar_div", &scalar_div, "Elementwise division",
          pybind11::arg("x"), pybind11::arg("divisor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
