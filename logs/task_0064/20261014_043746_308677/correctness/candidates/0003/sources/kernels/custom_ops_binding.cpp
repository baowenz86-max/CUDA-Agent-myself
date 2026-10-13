#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    const int64_t* in_strides,
    const int64_t* out_strides,
    cudaStream_t stream
);

extern "C" void nansum_dim12_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    const int64_t* in_strides,
    cudaStream_t stream
);

extern "C" void conv2d_launcher(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    int N, int C_in, int H_in, int W_in,
    int C_out, int KH, int KW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    int dilation_h, int dilation_w,
    const int64_t* in_strides,
    const int64_t* weight_strides,
    const int64_t* out_strides,
    cudaStream_t stream
);

// Helper: check CUDA error
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
}

// reflection_pad2d wrapper
torch::Tensor reflection_pad2d(
    torch::Tensor x,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N,C,H,W]");
    TORCH_CHECK(pad_left >= 0 && pad_right >= 0 && pad_top >= 0 && pad_bottom >= 0, "padding must be non-negative");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int out_H = H + pad_top + pad_bottom;
    int out_W = W + pad_left + pad_right;

    auto out = torch::empty({N, C, out_H, out_W}, x.options());

    // Prepare stride arrays (int64_t)
    std::vector<int64_t> in_strides(4);
    std::vector<int64_t> out_strides(4);
    for (int i = 0; i < 4; ++i) {
        in_strides[i] = x.stride(i);
        out_strides[i] = out.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        (int)pad_left, (int)pad_right, (int)pad_top, (int)pad_bottom,
        in_strides.data(),
        out_strides.data(),
        stream
    );
    CHECK_CUDA(cudaGetLastError());

    return out;
}

// conv2d wrapper using custom CUDA kernel
// Note: bias is required parameter, but may be None? The contract says bias is optional (nullable, optional: false means required parameter but can be null)
// Actually in contract, bias has optional: false, nullable: true, so it must be passed but can be None.
// In Python call, we pass self.conv.bias, which is not None. But handle nullptr.
torch::Tensor conv2d(
    torch::Tensor x,
    torch::Tensor weight,
    torch::optional<torch::Tensor> bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias must be float32");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 4 && weight.dim() == 4, "x and weight must be 4D");

    // Contract: weight shape [C_out, C_in, KH, KW]
    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int C_out = weight.size(0);
    int KH = weight.size(2);
    int KW = weight.size(3);

    TORCH_CHECK(weight.size(1) == C_in, "weight input channels mismatch");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    int H_out = (H_in + 2 * padding_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    int W_out = (W_in + 2 * padding_w - dilation_w * (KW - 1) - 1) / stride_w + 1;

    auto out = torch::empty({N, C_out, H_out, W_out}, x.options());

    // Prepare stride arrays
    std::vector<int64_t> in_strides(4), weight_strides(4), out_strides(4);
    for (int i = 0; i < 4; ++i) {
        in_strides[i] = x.stride(i);
        weight_strides[i] = weight.stride(i);
        out_strides[i] = out.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    conv2d_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        out.data_ptr<float>(),
        N, C_in, H_in, W_in,
        C_out, KH, KW,
        H_out, W_out,
        (int)stride_h, (int)stride_w,
        (int)padding_h, (int)padding_w,
        (int)dilation_h, (int)dilation_w,
        in_strides.data(),
        weight_strides.data(),
        out_strides.data(),
        stream
    );
    CHECK_CUDA(cudaGetLastError());

    return out;
}

// nansum_dim12 wrapper
torch::Tensor nansum_dim12(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D [N,C,H,W]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Output shape [N, W]
    auto out = torch::zeros({N, W}, x.options());

    std::vector<int64_t> in_strides(4);
    for (int i = 0; i < 4; ++i) {
        in_strides[i] = x.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    nansum_dim12_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        in_strides.data(),
        stream
    );
    CHECK_CUDA(cudaGetLastError());

    return out;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d", &reflection_pad2d, "ReflectionPad2d forward",
          py::arg("x"), py::arg("pad_left"), py::arg("pad_right"), py::arg("pad_top"), py::arg("pad_bottom"));
    m.def("conv2d", &conv2d, "Conv2d forward using custom CUDA kernel",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("stride_h"), py::arg("stride_w"),
          py::arg("padding_h"), py::arg("padding_w"), py::arg("dilation_h"), py::arg("dilation_w"));
    m.def("nansum_dim12", &nansum_dim12, "nansum over dims 1 and 2", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
