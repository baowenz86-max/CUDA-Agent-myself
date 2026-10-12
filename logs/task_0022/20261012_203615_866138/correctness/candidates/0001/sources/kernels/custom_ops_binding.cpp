// custom_ops_binding.cpp: PyTorch bindings for custom_ops.cu launchers
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void norm2_channel_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    cudaStream_t stream);

extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int Hin, int Win,
    int Hout, int Wout,
    int in_stride_n, int in_stride_c, int in_stride_h, int in_stride_w,
    int out_stride_n, int out_stride_c, int out_stride_h, int out_stride_w,
    bool align_corners,
    cudaStream_t stream);

// norm2_channel: L2 norm over channel dimension with keepdim=True.
// Input: [N, C, H, W] float32
// Output: [N, 1, H, W] float32
static torch::Tensor norm2_channel(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "norm2_channel: input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "norm2_channel: input must be 4D (N, C, H, W)");
    TORCH_CHECK(input.size(1) > 0, "norm2_channel: channel dimension must be positive");
    TORCH_CHECK(input.is_contiguous(), "norm2_channel: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble ||
                input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "norm2_channel: input dtype must be floating point");
    if (input.scalar_type() == at::kComplexFloat || input.scalar_type() == at::kComplexDouble) {
        TORCH_CHECK(false, "norm2_channel: complex input not supported");
    }
    // Convert to float32 for kernel execution if needed.
    torch::Tensor x = input;
    if (input.scalar_type() != at::kFloat) {
        x = input.to(torch::kFloat32);
    }
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    auto output = torch::empty({N, 1, H, W}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    norm2_channel_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)N, (int)C, (int)H, (int)W,
        (int)x.stride(0), (int)x.stride(1), (int)x.stride(2), (int)x.stride(3),
        (int)output.stride(0), (int)output.stride(1), (int)output.stride(2), (int)output.stride(3),
        stream);
    return output;
}

// upsample_bilinear2d: bilinear upsampling 2D.
// Input: [N, C, Hin, Win] float32
// Output: [N, C, Hout, Wout] float32
static torch::Tensor upsample_bilinear2d(
    torch::Tensor input,
    double scale_factor,
    bool align_corners) {
    TORCH_CHECK(input.is_cuda(), "upsample_bilinear2d: input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 4, "upsample_bilinear2d: input must be 4D (N, C, Hin, Win)");
    TORCH_CHECK(input.is_contiguous(), "upsample_bilinear2d: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kDouble ||
                input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16,
                "upsample_bilinear2d: input dtype must be floating point");
    TORCH_CHECK(scale_factor > 0, "upsample_bilinear2d: scale_factor must be positive");
    torch::Tensor x = input;
    if (input.scalar_type() != at::kFloat) {
        x = input.to(torch::kFloat32);
    }
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t Hin = x.size(2);
    int64_t Win = x.size(3);
    int64_t Hout = static_cast<int64_t>(std::floor(Hin * scale_factor));
    int64_t Wout = static_cast<int64_t>(std::floor(Win * scale_factor));
    TORCH_CHECK(Hout > 0 && Wout > 0, "upsample_bilinear2d: output size must be positive");
    auto output = torch::empty({N, C, Hout, Wout}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    upsample_bilinear2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)N, (int)C, (int)Hin, (int)Win,
        (int)Hout, (int)Wout,
        (int)x.stride(0), (int)x.stride(1), (int)x.stride(2), (int)x.stride(3),
        (int)output.stride(0), (int)output.stride(1), (int)output.stride(2), (int)output.stride(3),
        align_corners,
        stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("norm2_channel", &norm2_channel, "L2 norm over channel dimension (dim=1) keepdim=True",
          py::arg("input"));
    m.def("upsample_bilinear2d", &upsample_bilinear2d, "Bilinear 2D upsampling",
          py::arg("input"), py::arg("scale_factor"), py::arg("align_corners"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
