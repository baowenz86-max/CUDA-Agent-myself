#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_pad_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    float pad_value,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream);

extern "C" void custom_max_dim1_launcher(
    float* output,
    const float* input,
    int N, int H, int W,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream);

// Wrapper for custom_pad
torch::Tensor custom_pad(torch::Tensor input,
                         int64_t pad_left,
                         int64_t pad_right,
                         int64_t pad_top,
                         int64_t pad_bottom,
                         double value) {
    // Check tensor properties
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, C, H, W]");

    // Make input contiguous if needed (stride handling done via explicit strides in launcher,
    // but contiguous ensures better performance and simplifies pointer arithmetic)
    auto input_contig = input.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1);
    int H_in = input_contig.size(2);
    int W_in = input_contig.size(3);

    int H_out = H_in + pad_top + pad_bottom;
    int W_out = W_in + pad_left + pad_right;

    // Allocate output tensor
    auto output = torch::empty({N, C, H_out, W_out}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Use contiguous strides for both input and output (output is contiguous by creation)
    long long input_n_stride = input_contig.stride(0);
    long long input_c_stride = input_contig.stride(1);
    long long input_h_stride = input_contig.stride(2);
    long long input_w_stride = input_contig.stride(3);

    // Call launcher
    custom_pad_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        N, C, H_in, W_in,
        H_out, W_out,
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        static_cast<float>(value),
        input_n_stride, input_c_stride, input_h_stride, input_w_stride,
        stream);

    return output;
}

// Wrapper for custom_max_dim1
torch::Tensor custom_max_dim1(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D [N, C, H, W]");
    TORCH_CHECK(input.size(1) == 1, "dim=1 size must be 1 for current implementation");

    // Make input contiguous (launcher uses strides, but contiguous ensures correctness)
    auto input_contig = input.contiguous();

    int N = input_contig.size(0);
    int C = input_contig.size(1); // should be 1
    int H = input_contig.size(2);
    int W = input_contig.size(3);

    // Allocate output tensor [N, H, W]
    auto output = torch::empty({N, H, W}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long input_n_stride = input_contig.stride(0);
    long long input_c_stride = input_contig.stride(1);
    long long input_h_stride = input_contig.stride(2);
    long long input_w_stride = input_contig.stride(3);

    custom_max_dim1_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        N, H, W,
        input_n_stride, input_c_stride, input_h_stride, input_w_stride,
        stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_pad", &custom_pad,
          "ConstantPad2d forward (CUDA)",
          py::arg("input"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"),
          py::arg("value"));
    m.def("custom_max_dim1", &custom_max_dim1,
          "Max over dim=1 (size 1) forward (CUDA)",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
