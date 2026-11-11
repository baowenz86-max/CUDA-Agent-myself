// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

extern "C" void tanh_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream);

extern "C" void conv3d_forward_launcher(const float* input,
                                        const float* weight,
                                        const float* bias,
                                        float* output,
                                        int N, int C_in, int D_in, int H_in, int W_in,
                                        int C_out, int KD, int KH, int KW,
                                        int D_out, int H_out, int W_out,
                                        bool has_bias,
                                        cudaStream_t stream);

namespace {

std::vector<int> get_3d_param(const pybind11::object& obj, const char* param_name) {
    std::vector<int> result;
    if (pybind11::isinstance<pybind11::int_>(obj)) {
        int val = obj.cast<int>();
        result = {val, val, val};
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        auto t = obj.cast<pybind11::tuple>();
        if (t.size() != 3)
            throw std::runtime_error(std::string(param_name) + " must be a 3-tuple or int");
        for (auto item : t) result.push_back(item.cast<int>());
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        auto l = obj.cast<pybind11::list>();
        if (l.size() != 3)
            throw std::runtime_error(std::string(param_name) + " must be a 3-list or int");
        for (auto item : l) result.push_back(item.cast<int>());
    } else {
        throw std::runtime_error(std::string(param_name) + " must be int, tuple, or list");
    }
    return result;
}

} // namespace

torch::Tensor conv3d_forward(torch::Tensor input,
                             torch::Tensor weight,
                             c10::optional<torch::Tensor> bias,
                             pybind11::object stride,
                             pybind11::object padding,
                             pybind11::object dilation) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    bool has_bias = bias.has_value();
    if (has_bias) {
        bias_contig = bias->contiguous();
        TORCH_CHECK(bias_contig.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias_contig.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias_contig.dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias_contig.size(0) == weight_contig.size(0), "bias size mismatch");
    }

    std::vector<int> stride_vec = get_3d_param(stride, "stride");
    std::vector<int> padding_vec = get_3d_param(padding, "padding");
    std::vector<int> dilation_vec = get_3d_param(dilation, "dilation");

    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK(stride_vec[i] >= 1, "stride must be >=1");
        TORCH_CHECK(padding_vec[i] >= 0, "padding must be >=0");
        TORCH_CHECK(dilation_vec[i] >= 1, "dilation must be >=1");
    }

    const int N = input_contig.size(0);
    const int C_in = input_contig.size(1);
    const int D_in = input_contig.size(2);
    const int H_in = input_contig.size(3);
    const int W_in = input_contig.size(4);
    const int C_out = weight_contig.size(0);
    const int C_in_weight = weight_contig.size(1);
    const int KD = weight_contig.size(2);
    const int KH = weight_contig.size(3);
    const int KW = weight_contig.size(4);

    TORCH_CHECK(C_in == C_in_weight, "input channels mismatch");

    auto calc_out_dim = [](int in_dim, int pad, int dilation_, int kernel, int stride_) {
        return (in_dim + 2 * pad - dilation_ * (kernel - 1) - 1) / stride_ + 1;
    };
    const int D_out = calc_out_dim(D_in, padding_vec[0], dilation_vec[0], KD, stride_vec[0]);
    const int H_out = calc_out_dim(H_in, padding_vec[1], dilation_vec[1], KH, stride_vec[1]);
    const int W_out = calc_out_dim(W_in, padding_vec[2], dilation_vec[2], KW, stride_vec[2]);

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "output dims must be positive");

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // For simplicity, only support stride=1, padding=0, dilation=1 (as per task contract)
    TORCH_CHECK(stride_vec[0] == 1 && stride_vec[1] == 1 && stride_vec[2] == 1,
                "This implementation only supports stride=1");
    TORCH_CHECK(padding_vec[0] == 0 && padding_vec[1] == 0 && padding_vec[2] == 0,
                "This implementation only supports padding=0");
    TORCH_CHECK(dilation_vec[0] == 1 && dilation_vec[1] == 1 && dilation_vec[2] == 1,
                "This implementation only supports dilation=1");

    conv3d_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        has_bias ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, C_in, D_in, H_in, W_in,
        C_out, KD, KH, KW,
        D_out, H_out, W_out,
        has_bias,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv3d_forward CUDA launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor tanh_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    if (numel == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tanh_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tanh_forward CUDA launch failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv3d_forward", &conv3d_forward,
          "3D convolution forward",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("dilation") = 1);
    m.def("tanh_forward", &tanh_forward,
          "Element-wise tanh forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);