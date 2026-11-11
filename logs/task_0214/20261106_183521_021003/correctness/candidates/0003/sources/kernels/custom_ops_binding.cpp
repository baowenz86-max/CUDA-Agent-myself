// PyTorch binding for custom operators
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad,
    cudaStream_t stream);

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void conv2d_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    int K, int R, int S,
    int outH, int outW,
    cudaStream_t stream);

static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

static void check_cuda_float32(const torch::Tensor& tensor, const std::string& name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    check_cuda_float32(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int pad = static_cast<int>(padding);
    int outH = H + 2 * pad;
    int outW = W + 2 * pad;

    auto output = torch::empty({N, C, outH, outW}, input.options());

    cudaStream_t stream = getCurrentStream();
    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        pad,
        stream);

    return output;
}

torch::Tensor sigmoid_forward(torch::Tensor input) {
    check_cuda_float32(input, "input");
    TORCH_CHECK(input.dim() >= 1, "input must have at least 1 dimension");

    auto output = torch::empty_like(input);
    int total = static_cast<int>(input.numel());

    cudaStream_t stream = getCurrentStream();
    sigmoid_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        stream);

    return output;
}

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    check_cuda_float32(input, "input");
    check_cuda_float32(weight, "weight");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    // Task-specific: stride=1, padding=0, dilation=1, groups=1
    TORCH_CHECK(stride == 1 && padding == 0 && dilation == 1 && groups == 1,
                "Only stride=1, padding=0, dilation=1, groups=1 supported");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int K = weight.size(0);
    int C_per_group = weight.size(1);
    int R = weight.size(2);
    int S = weight.size(3);

    TORCH_CHECK(C == C_per_group, "input channels must equal weight[1] for groups=1");

    int outH = H - R + 1;
    int outW = W - S + 1;

    auto output = torch::empty({N, K, outH, outW}, input.options());

    cudaStream_t stream = getCurrentStream();
    conv2d_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, K, R, S, outH, outW,
        stream);

    return output;
}

torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float32(input, "input");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D (C,1,1)");
    TORCH_CHECK(bias.size(0) == input.size(1), "bias channel must match input channels");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias spatial dims must be 1");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = getCurrentStream();
    add_bias_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection pad 2d forward",
          py::arg("input"), py::arg("padding"));
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid forward",
          py::arg("input"));
    m.def("conv2d_forward", &conv2d_forward,
          "Conv2d forward (naive CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"), py::arg("groups"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias forward",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
