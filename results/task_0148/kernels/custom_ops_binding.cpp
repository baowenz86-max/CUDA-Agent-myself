#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reflection_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int out_H, int out_W,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
    cudaStream_t stream);

extern "C" void softmax_dim1_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int in_n_stride, int in_c_stride, int in_h_stride, int in_w_stride,
    int out_n_stride, int out_c_stride, int out_h_stride, int out_w_stride,
    cudaStream_t stream);

extern "C" void histc_launcher(
    const float* input,
    float* output,
    int numel,
    int bins,
    float min_val,
    float max_val,
    int n_stride, int c_stride, int h_stride, int w_stride,
    int N, int C, int H, int W,
    cudaStream_t stream);

// Wrapper functions

torch::Tensor reflection_pad2d_forward(torch::Tensor input, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(padding == 1, "Only padding=1 supported for reflection pad");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int out_H = H + 2 * padding;
    int out_W = W + 2 * padding;

    auto output = torch::empty({N, C, out_H, out_W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass strides explicitly to handle non-contiguous inputs.
    // If input is contiguous, strides are as expected.
    int in_n_stride = input.stride(0);
    int in_c_stride = input.stride(1);
    int in_h_stride = input.stride(2);
    int in_w_stride = input.stride(3);
    int out_n_stride = output.stride(0);
    int out_c_stride = output.stride(1);
    int out_h_stride = output.stride(2);
    int out_w_stride = output.stride(3);

    reflection_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        out_H, out_W,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "reflection_pad2d_launcher failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor softmax_dim1_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int in_n_stride = input.stride(0);
    int in_c_stride = input.stride(1);
    int in_h_stride = input.stride(2);
    int in_w_stride = input.stride(3);
    int out_n_stride = output.stride(0);
    int out_c_stride = output.stride(1);
    int out_h_stride = output.stride(2);
    int out_w_stride = output.stride(3);

    softmax_dim1_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        in_n_stride, in_c_stride, in_h_stride, in_w_stride,
        out_n_stride, out_c_stride, out_h_stride, out_w_stride,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax_dim1_launcher failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor histc_forward(torch::Tensor input, int64_t bins, double min, double max) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(bins > 0, "bins must be positive");
    TORCH_CHECK(min < max, "min must be less than max");

    auto output = torch::zeros({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    int numel = N * C * H * W;
    int n_stride = input.stride(0);
    int c_stride = input.stride(1);
    int h_stride = input.stride(2);
    int w_stride = input.stride(3);

    histc_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        static_cast<int>(bins),
        static_cast<float>(min),
        static_cast<float>(max),
        n_stride, c_stride, h_stride, w_stride,
        N, C, H, W,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "histc_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward,
          "Reflection pad 2d forward",
          py::arg("input"), py::arg("padding"));
    m.def("softmax_dim1_forward", &softmax_dim1_forward,
          "Softmax over channel dimension forward",
          py::arg("input"));
    m.def("histc_forward", &histc_forward,
          "Histogram forward",
          py::arg("input"), py::arg("bins"), py::arg("min"), py::arg("max"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
