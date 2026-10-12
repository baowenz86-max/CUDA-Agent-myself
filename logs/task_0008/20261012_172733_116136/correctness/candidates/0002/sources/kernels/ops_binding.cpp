#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare CUDA launchers from ops.cu
extern "C" void pad_constant_2d_launcher(
    const float* input, float* output,
    int batch, int channels, int in_h, int in_w,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    float pad_value, cudaStream_t stream);
extern "C" void rot90_k1_launcher(
    const float* input, float* output,
    int batch, int channels, int height, int width,
    cudaStream_t stream);
extern "C" void conv2d_forward_launcher(
    const float* input, const float* weight, const float* bias, float* output,
    int N, int C, int H, int W, int K, int R, int S,
    int OH, int OW, cudaStream_t stream);
extern "C" void isinf_launcher(
    const float* input, bool* output, int64_t total, cudaStream_t stream);
extern "C" void masked_fill_zero_launcher(
    float* data, const bool* mask, int64_t total, cudaStream_t stream);
extern "C" void add_bias_launcher(
    float* data, const float* bias, int K, int H, int W, int64_t total, cudaStream_t stream);

// PyTorch binding: pad_constant_2d
torch::Tensor pad_constant_2d(torch::Tensor input, std::vector<int64_t> padding, double value) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "expected 4D tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(padding.size() == 4, "padding must have 4 elements");

    int pad_left = padding[0];
    int pad_right = padding[1];
    int pad_top = padding[2];
    int pad_bottom = padding[3];

    auto sizes = input.sizes();
    int64_t out_h = sizes[2] + pad_top + pad_bottom;
    int64_t out_w = sizes[3] + pad_left + pad_right;
    auto output = torch::empty({sizes[0], sizes[1], out_h, out_w}, input.options());
    if (output.numel() == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    pad_constant_2d_launcher(
        input.data_ptr<float>(), output.data_ptr<float>(),
        input.size(0), input.size(1), input.size(2), input.size(3),
        pad_top, pad_bottom, pad_left, pad_right,
        (float)value, stream);
    return output;
}

// PyTorch binding: rot90_k1
torch::Tensor rot90_k1(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dim() == 4, "expected 4D tensor");

    auto sizes = input.sizes();
    int64_t out_h = sizes[3];
    int64_t out_w = sizes[2];
    auto output = torch::empty({sizes[0], sizes[1], out_h, out_w}, input.options());
    if (output.numel() == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    rot90_k1_launcher(
        input.data_ptr<float>(), output.data_ptr<float>(),
        input.size(0), input.size(1), input.size(2), input.size(3), stream);
    return output;
}

// PyTorch binding: conv2d_forward
torch::Tensor conv2d_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && weight.is_cuda() && bias.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && weight.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "must be float32");
    TORCH_CHECK(input.is_contiguous() && weight.is_contiguous() && bias.is_contiguous(), "contiguous required");
    TORCH_CHECK(input.dim() == 4 && weight.dim() == 4, "expected 4D tensors");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int K = weight.size(0);
    int R = weight.size(2);
    int S = weight.size(3);

    TORCH_CHECK(weight.size(1) == C, "input channels mismatch");
    TORCH_CHECK(bias.numel() == K, "bias size mismatch");

    int OH = H - R + 1;
    int OW = W - S + 1;
    auto output = torch::empty({N, K, OH, OW}, input.options());
    if (output.numel() == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv2d_forward_launcher(
        input.data_ptr<float>(), weight.data_ptr<float>(), bias.data_ptr<float>(), output.data_ptr<float>(),
        N, C, H, W, K, R, S, OH, OW, stream);
    return output;
}

// PyTorch binding: isinf
torch::Tensor isinf_custom(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty(input.sizes(), input.options().dtype(torch::kBool));
    if (output.numel() == 0) return output;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    isinf_launcher(input.data_ptr<float>(), output.data_ptr<bool>(), input.numel(), stream);
    return output;
}

// PyTorch binding: masked_fill_zero
torch::Tensor masked_fill_zero(torch::Tensor input, torch::Tensor mask) {
    TORCH_CHECK(input.is_cuda() && mask.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && mask.dtype() == torch::kBool, "dtype mismatch");
    TORCH_CHECK(input.sizes() == mask.sizes(), "size mismatch");
    TORCH_CHECK(input.is_contiguous() && mask.is_contiguous(), "contiguous required");

    if (input.numel() == 0) return input;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    masked_fill_zero_launcher(input.data_ptr<float>(), mask.data_ptr<bool>(), input.numel(), stream);
    return input;
}

// PyTorch binding: add_bias
torch::Tensor add_bias(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda() && bias.is_cuda(), "tensors must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32 && bias.dtype() == torch::kFloat32, "dtype mismatch");
    TORCH_CHECK(input.dim() == 4 && bias.dim() == 3, "dim mismatch");
    TORCH_CHECK(bias.size(0) == input.size(1), "bias channel mismatch");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias must be [K,1,1]");
    TORCH_CHECK(input.is_contiguous() && bias.is_contiguous(), "contiguous required");

    if (input.numel() == 0) return input;

    int K = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(input.data_ptr<float>(), bias.data_ptr<float>(), K, H, W, input.numel(), stream);
    return input;
}

// Register all bindings
void register_ops_binding(pybind11::module& m) {
    m.def("pad_constant_2d", &pad_constant_2d, "ConstantPad2d");
    m.def("rot90_k1", &rot90_k1, "Rot90 with k=1 on dims 2,3");
    m.def("conv2d_forward", &conv2d_forward, "Conv2d forward");
    m.def("isinf", &isinf_custom, "Elementwise isinf");
    m.def("masked_fill_zero", &masked_fill_zero, "Masked fill with zero");
    m.def("add_bias", &add_bias, "Add bias with broadcasting");
}

REGISTER_BINDING(ops_binding, register_ops_binding);
