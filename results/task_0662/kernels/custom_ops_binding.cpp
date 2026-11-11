#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_pad2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int pad_left, int pad_right, int pad_top, int pad_bottom,
    long long input_n_stride, long long input_c_stride,
    long long input_h_stride, long long input_w_stride,
    cudaStream_t stream);

extern "C" void custom_mean_dim_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int dim,
    cudaStream_t stream);

// Wrapper for custom_pad2d
torch::Tensor custom_pad2d_forward(
    torch::Tensor input,
    int64_t pad_left,
    int64_t pad_right,
    int64_t pad_top,
    int64_t pad_bottom)
{
    // Check input tensor
    TORCH_CHECK(input.is_cuda(), "custom_pad2d: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_pad2d: input must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_pad2d: input must be 4D [N,C,H,W]");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int H_out = H + static_cast<int>(pad_top + pad_bottom);
    int W_out = W + static_cast<int>(pad_left + pad_right);

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    // Get input strides (in elements)
    long long input_n_stride = input.stride(0);
    long long input_c_stride = input.stride(1);
    long long input_h_stride = input.stride(2);
    long long input_w_stride = input.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    custom_pad2d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(pad_left),
        static_cast<int>(pad_right),
        static_cast<int>(pad_top),
        static_cast<int>(pad_bottom),
        input_n_stride,
        input_c_stride,
        input_h_stride,
        input_w_stride,
        stream
    );

    return output;
}

// Wrapper for custom_mean_dim
torch::Tensor custom_mean_dim_forward(
    torch::Tensor input,
    int64_t dim)
{
    // Check input tensor
    TORCH_CHECK(input.is_cuda(), "custom_mean_dim: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_mean_dim: input must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_mean_dim: input must be 4D [N,C,H,W]");

    // The .cu only supports dim=2
    TORCH_CHECK(dim == 2, "custom_mean_dim: only dim=2 is supported");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Output shape: remove dimension 2
    auto output = torch::empty({N, C, W}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher (contiguous input expected; output contiguous)
    custom_mean_dim_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(dim),
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_pad2d", &custom_pad2d_forward,
          "Apply ZeroPad2d padding",
          py::arg("input"),
          py::arg("pad_left"),
          py::arg("pad_right"),
          py::arg("pad_top"),
          py::arg("pad_bottom"));

    m.def("custom_mean_dim", &custom_mean_dim_forward,
          "Compute mean over dimension",
          py::arg("input"),
          py::arg("dim"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
