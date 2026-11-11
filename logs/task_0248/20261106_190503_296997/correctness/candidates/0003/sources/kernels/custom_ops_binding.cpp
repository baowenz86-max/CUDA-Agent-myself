// kernels/custom_ops_binding.cpp

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
    void gelu_forward_launcher(
        const float* input,
        float* output,
        int64_t numel,
        int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4,
        int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
        int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
        cudaStream_t stream);

    void bias_add_forward_launcher(
        const float* input,
        const float* bias,
        float* output,
        int64_t numel,
        int64_t C,
        int64_t D, int64_t H, int64_t W, int64_t N,
        int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
        int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
        cudaStream_t stream);
}

// Helper to parse int or tuple of 3 ints
static std::vector<int> parse_3_int(const pybind11::handle& obj, const char* name) {
    std::vector<int> result(3);
    if (pybind11::isinstance<pybind11::int_>(obj)) {
        int v = obj.cast<int>();
        result = {v, v, v};
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        auto tup = obj.cast<pybind11::tuple>();
        TORCH_CHECK(tup.size() == 3, name, " must be int or tuple of 3 ints");
        for (int i = 0; i < 3; ++i) {
            result[i] = tup[i].cast<int>();
        }
    } else {
        TORCH_CHECK(false, name, " must be int or tuple of 3 ints");
    }
    return result;
}

// conv3d_forward placeholder: we need to implement convolution. Since cuDNN headers are not available,
// we will use a naive CPU implementation? No, must be CUDA. We'll implement a simple direct convolution
// kernel in the .cu later, but for now, we can't include cuDNN. This binding will call a custom conv launcher
// that we will add to custom_ops.cu. But to keep this file compilable, we'll declare the launcher and
// implement it in a separate step. However, we must provide a definition now to avoid link errors.
// We'll add a minimal conv3d implementation in custom_ops.cu using a generic kernel.
// But since we are only modifying custom_ops.cu and binding, we can add the kernel and launcher there.
// For now, declare the conv3d launcher and implement it later in custom_ops.cu.
// To avoid missing symbol, we'll implement a simple one.

extern "C" void conv3d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int D, int H, int W,
    int C_out,
    int kD, int kH, int kW,
    int strideD, int strideH, int strideW,
    int padD, int padH, int padW,
    int D_out, int H_out, int W_out,
    cudaStream_t stream);

// conv3d_forward wrapper
static torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    pybind11::object stride,
    pybind11::object padding) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(weight.size(1) == input.size(1), "input/weight channel mismatch");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight/bias channel mismatch");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    std::vector<int> stride_arr = parse_3_int(stride, "stride");
    std::vector<int> padding_arr = parse_3_int(padding, "padding");
    const int strideD = stride_arr[0], strideH = stride_arr[1], strideW = stride_arr[2];
    const int padD = padding_arr[0], padH = padding_arr[1], padW = padding_arr[2];

    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int C_out = weight.size(0);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int D_out = (D + 2 * padD - kD) / strideD + 1;
    const int H_out = (H + 2 * padH - kH) / strideH + 1;
    const int W_out = (W + 2 * padW - kW) / strideW + 1;

    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv3d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C_in, D, H, W,
        C_out,
        kD, kH, kW,
        strideD, strideH, strideW,
        padD, padH, padW,
        D_out, H_out, W_out,
        stream
    );

    return output;
}

// gelu_forward wrapper
static torch::Tensor gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);

    int64_t numel = input.numel();
    if (numel == 0) return output;

    if (input.dim() != 5) {
        TORCH_CHECK(false, "gelu_forward expects 5D tensor as per contract");
    }

    int64_t dims[5], in_strides[5], out_strides[5];
    for (int i = 0; i < 5; ++i) {
        dims[i] = input.size(i);
        in_strides[i] = input.stride(i);
        out_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    gelu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        dims[0], dims[1], dims[2], dims[3], dims[4],
        in_strides[0], in_strides[1], in_strides[2], in_strides[3], in_strides[4],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
        stream
    );

    return output;
}

// bias_add_forward wrapper
static torch::Tensor bias_add_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(bias.numel() == input.size(1), "bias size must match channel dim");

    auto output = torch::empty_like(input);

    int64_t numel = input.numel();
    if (numel == 0) return output;

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t D = input.size(2);
    const int64_t H = input.size(3);
    const int64_t W = input.size(4);

    int64_t in_strides[5], out_strides[5];
    for (int i = 0; i < 5; ++i) {
        in_strides[i] = input.stride(i);
        out_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_forward_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        C,
        D, H, W, N,
        in_strides[0], in_strides[1], in_strides[2], in_strides[3], in_strides[4],
        out_strides[0], out_strides[1], out_strides[2], out_strides[3], out_strides[4],
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("conv3d_forward", &conv3d_forward,
          "Conv3d forward using custom CUDA kernel",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"));
    m.def("gelu_forward", &gelu_forward,
          "GELU activation forward",
          pybind11::arg("input"));
    m.def("bias_add_forward", &bias_add_forward,
          "Bias add forward",
          pybind11::arg("input"),
          pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
