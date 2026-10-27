#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_features,
    int spatial0,
    int spatial1,
    int out_features,
    long long input_stride_batch,
    long long input_stride_channel,
    long long input_stride_spatial0,
    long long input_stride_spatial1,
    long long weight_stride_out,
    long long weight_stride_in,
    long long output_stride_batch,
    long long output_stride_channel,
    long long output_stride_spatial0,
    long long output_stride_out,
    cudaStream_t stream
);

extern "C" void custom_silu_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channel,
    int spatial0,
    int spatial1,
    long long stride_batch,
    long long stride_channel,
    long long stride_spatial0,
    long long stride_spatial1,
    cudaStream_t stream
);

extern "C" void custom_permute_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channel,
    int spatial0,
    int spatial1,
    long long input_stride_batch,
    long long input_stride_channel,
    long long input_stride_spatial0,
    long long input_stride_spatial1,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Custom linear forward
static torch::Tensor custom_linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    TORCH_CHECK(input.is_cuda(), "custom_linear_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "custom_linear_forward: weight must be a CUDA tensor");
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().is_cuda(), "custom_linear_forward: bias must be a CUDA tensor");
    }

    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_linear_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "custom_linear_forward: weight must be float32");
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        TORCH_CHECK(bias_opt.value().dtype() == torch::kFloat32, "custom_linear_forward: bias must be float32");
    }

    TORCH_CHECK(input.dim() == 4, "custom_linear_forward: input must be 4D (batch, in_features, spatial0, spatial1)");
    TORCH_CHECK(weight.dim() == 2, "custom_linear_forward: weight must be 2D (out_features, in_features)");
    int batch = input.size(0);
    int in_features = input.size(1);
    int spatial0 = input.size(2);
    int spatial1 = input.size(3);
    int out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "custom_linear_forward: weight.shape[1] must equal input.shape[1]");

    torch::Tensor bias;
    bool has_bias = bias_opt.has_value() && bias_opt.value().defined();
    if (has_bias) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features, "custom_linear_forward: bias shape must be [out_features]");
    }

    torch::Tensor input_contig = input.contiguous();
    torch::Tensor weight_contig = weight.contiguous();

    auto output = torch::empty({batch, in_features, spatial0, out_features}, input.options());

    long long in_stride_b = input_contig.stride(0);
    long long in_stride_c = input_contig.stride(1);
    long long in_stride_h = input_contig.stride(2);
    long long in_stride_w = input_contig.stride(3);
    long long w_stride_o = weight_contig.stride(0);
    long long w_stride_i = weight_contig.stride(1);
    long long out_stride_b = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_o = output.stride(3);

    cudaStream_t stream = getCurrentStream();
    const float* bias_ptr = has_bias ? bias.data_ptr<float>() : nullptr;

    custom_linear_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch, in_features, spatial0, spatial1, out_features,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        w_stride_o, w_stride_i,
        out_stride_b, out_stride_c, out_stride_h, out_stride_o,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_linear_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Custom SiLU forward
static torch::Tensor custom_silu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "custom_silu_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_silu_forward: input must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_silu_forward: input must be 4D");

    int batch = input.size(0);
    int channel = input.size(1);
    int spatial0 = input.size(2);
    int spatial1 = input.size(3);

    torch::Tensor input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    long long stride_b = input_contig.stride(0);
    long long stride_c = input_contig.stride(1);
    long long stride_h = input_contig.stride(2);
    long long stride_w = input_contig.stride(3);

    cudaStream_t stream = getCurrentStream();

    custom_silu_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channel, spatial0, spatial1,
        stride_b, stride_c, stride_h, stride_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_silu_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Custom permute forward: fixed permutation (0,2,3,1)
// The dims parameter is required as an int per contract, but the actual permutation is fixed.
// We ignore the value and always perform (0,2,3,1).
static torch::Tensor custom_permute_forward(torch::Tensor input, int64_t dims) {
    TORCH_CHECK(input.is_cuda(), "custom_permute_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_permute_forward: input must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_permute_forward: input must be 4D");

    int batch = input.size(0);
    int channel = input.size(1);
    int spatial0 = input.size(2);
    int spatial1 = input.size(3);

    torch::Tensor input_contig = input.contiguous();
    auto output = torch::empty({batch, spatial0, spatial1, channel}, input.options());

    long long stride_b = input_contig.stride(0);
    long long stride_c = input_contig.stride(1);
    long long stride_h = input_contig.stride(2);
    long long stride_w = input_contig.stride(3);

    cudaStream_t stream = getCurrentStream();

    custom_permute_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channel, spatial0, spatial1,
        stride_b, stride_c, stride_h, stride_w,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_permute_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("custom_linear_forward", &custom_linear_forward,
          "Custom linear forward",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = pybind11::none());
    m.def("custom_silu_forward", &custom_silu_forward,
          "Custom SiLU forward",
          pybind11::arg("input"));
    m.def("custom_permute_forward", &custom_permute_forward,
          "Custom permute forward (fixed (0,2,3,1))",
          pybind11::arg("input"),
          pybind11::arg("dims"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
