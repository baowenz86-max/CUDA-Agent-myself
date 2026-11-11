#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void clamp_forward_launcher(
    const float* input,
    float* output,
    float min_val,
    float max_val,
    long long numel,
    cudaStream_t stream);

extern "C" void adaptive_avg_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    long long stride_N, long long stride_C, long long stride_H, long long stride_W,
    long long out_stride_N, long long out_stride_C, long long out_stride_H, long long out_stride_W,
    cudaStream_t stream);

// Wrapper for clamp
torch::Tensor clamp_forward(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    // clamps supports any strides; no contiguous requirement
    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        x.numel(),
        stream);

    return output;
}

// Wrapper for adaptive average pooling
// output_size is a tuple of two ints; we use py::tuple to accept it and extract ints.
torch::Tensor adaptive_avg_pool2d_forward(torch::Tensor x, py::tuple output_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "adaptive_avg_pool2d expects 4D input (N,C,H,W)");

    // Extract output sizes
    if (py::len(output_size) != 2) {
        throw std::invalid_argument("output_size must be a tuple of two ints");
    }
    int H_out = output_size[0].cast<int>();
    int W_out = output_size[1].cast<int>();
    TORCH_CHECK(H_out > 0 && W_out > 0, "output_size values must be positive");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // Prepare output tensor with shape (N, C, H_out, W_out) and float32
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Compute input strides (in elements); x may be non-contiguous
    long long stride_N = x.stride(0);
    long long stride_C = x.stride(1);
    long long stride_H = x.stride(2);
    long long stride_W = x.stride(3);

    // Output is contiguous, so strides are in standard order
    long long out_stride_N = output.stride(0);
    long long out_stride_C = output.stride(1);
    long long out_stride_H = output.stride(2);
    long long out_stride_W = output.stride(3);

    adaptive_avg_pool2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        stride_N, stride_C, stride_H, stride_W,
        out_stride_N, out_stride_C, out_stride_H, out_stride_W,
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("clamp_forward", &clamp_forward,
          "Clamp tensor to [min_val, max_val]",
          py::arg("x"), py::arg("min_val"), py::arg("max_val"));
    m.def("adaptive_avg_pool2d_forward", &adaptive_avg_pool2d_forward,
          "Adaptive average pooling 2D",
          py::arg("x"), py::arg("output_size"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
