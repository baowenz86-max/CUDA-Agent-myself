#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void zero_pad_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    int pad_top, int pad_bottom, int pad_left, int pad_right,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream);

torch::Tensor zero_pad_forward(
    torch::Tensor x,
    int64_t pad_top,
    int64_t pad_bottom,
    int64_t pad_left,
    int64_t pad_right) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    // Padding values are non-negative
    TORCH_CHECK(pad_top >= 0 && pad_bottom >= 0 && pad_left >= 0 && pad_right >= 0,
                "Padding must be non-negative");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);

    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    // Output shape and strides (contiguous)
    std::vector<int64_t> out_shape = {N, C, H_out, W_out};
    std::vector<int64_t> out_strides = {
        C * H_out * W_out,
        H_out * W_out,
        W_out,
        1
    };

    auto output = torch::empty_strided(out_shape, out_strides, x.options());

    // Input strides may be non-contiguous; pass as long long
    auto input_strides = x.strides();
    long long input_n_stride = static_cast<long long>(input_strides[0]);
    long long input_c_stride = static_cast<long long>(input_strides[1]);
    long long input_h_stride = static_cast<long long>(input_strides[2]);
    long long input_w_stride = static_cast<long long>(input_strides[3]);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    zero_pad_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H_in),
        static_cast<int>(W_in),
        static_cast<int>(H_out),
        static_cast<int>(W_out),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        input_n_stride, input_c_stride, input_h_stride, input_w_stride,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_zero_pad(pybind11::module& m) {
    m.def("zero_pad_forward", &zero_pad_forward,
          "Zero padding forward (CUDA)",
          py::arg("x"),
          py::arg("pad_top"),
          py::arg("pad_bottom"),
          py::arg("pad_left"),
          py::arg("pad_right"));
}

REGISTER_BINDING(zero_pad, register_zero_pad);
