#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batchnorm_eval_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int HW,
    cudaStream_t stream
);

extern "C" void bmm_launcher(
    float* output,
    const float* mat1,
    const float* mat2,
    int batch,
    int M,
    int N,
    int K,
    cudaStream_t stream
);

extern "C" void conv_transpose2d_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int batch,
    int in_channels,
    int out_channels,
    int kernel_H,
    int kernel_W,
    int stride_H,
    int stride_W,
    int padding_H,
    int padding_W,
    int output_padding_H,
    int output_padding_W,
    int seq_len,
    cudaStream_t stream
);

// Helper to get current CUDA stream
static inline cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// BatchNorm eval forward wrapper
torch::Tensor batchnorm_eval_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dtype() == torch::kFloat32,
                "weight must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.dtype() == torch::kFloat32,
                "bias must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.is_contiguous() && running_mean.dtype() == torch::kFloat32,
                "running_mean must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(running_var.is_cuda() && running_var.is_contiguous() && running_var.dtype() == torch::kFloat32,
                "running_var must be a contiguous float32 CUDA tensor");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    int N = input.size(0);
    int C = input.size(1);
    int HW = input.size(2) * input.size(3);
    TORCH_CHECK(weight.numel() == C, "weight size must match channels");
    TORCH_CHECK(bias.numel() == C, "bias size must match channels");
    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match channels");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match channels");

    auto output = torch::empty_like(input);

    float eps_float = static_cast<float>(eps);

    batchnorm_eval_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps_float,
        input.numel(),
        C,
        HW,
        getCurrentCUDAStream()
    );

    return output;
}

// BMM forward wrapper
torch::Tensor bmm_forward(torch::Tensor mat1, torch::Tensor mat2) {
    TORCH_CHECK(mat1.is_cuda() && mat1.is_contiguous() && mat1.dtype() == torch::kFloat32,
                "mat1 must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(mat2.is_cuda() && mat2.is_contiguous() && mat2.dtype() == torch::kFloat32,
                "mat2 must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(mat1.dim() == 3, "mat1 must be 3D (batch, M, K)");
    TORCH_CHECK(mat2.dim() == 3, "mat2 must be 3D (batch, K, N)");
    TORCH_CHECK(mat1.size(0) == mat2.size(0), "batch sizes must match");
    TORCH_CHECK(mat1.size(2) == mat2.size(1), "inner dimensions must match");

    int batch = mat1.size(0);
    int M = mat1.size(1);
    int N = mat2.size(2);
    int K = mat1.size(2);

    auto output = torch::empty({batch, M, N}, mat1.options());

    bmm_launcher(
        output.data_ptr<float>(),
        mat1.data_ptr<float>(),
        mat2.data_ptr<float>(),
        batch,
        M,
        N,
        K,
        getCurrentCUDAStream()
    );

    return output;
}

// ConvTranspose2d forward wrapper
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t batch,
    int64_t in_channels,
    int64_t out_channels,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t output_padding_h,
    int64_t output_padding_w,
    int64_t seq_len
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.dtype() == torch::kFloat32,
                "input must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.dtype() == torch::kFloat32,
                "weight must be a contiguous float32 CUDA tensor");
    TORCH_CHECK(bias.is_cuda() && bias.is_contiguous() && bias.dtype() == torch::kFloat32,
                "bias must be a contiguous float32 CUDA tensor");

    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, seq_len) for this custom conv");

    // The launcher expects input as (batch, in_channels, 1, seq_len)
    int in_H = 1;
    int in_W = seq_len;
    int out_H = (in_H - 1) * stride_h - 2 * padding_h + kernel_h + output_padding_h;
    int out_W = (in_W - 1) * stride_w - 2 * padding_w + kernel_w + output_padding_w;

    auto output = torch::empty({batch, out_channels, out_H, out_W}, input.options());

    conv_transpose2d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(kernel_h),
        static_cast<int>(kernel_w),
        static_cast<int>(stride_h),
        static_cast<int>(stride_w),
        static_cast<int>(padding_h),
        static_cast<int>(padding_w),
        static_cast<int>(output_padding_h),
        static_cast<int>(output_padding_w),
        static_cast<int>(seq_len),
        getCurrentCUDAStream()
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("batchnorm_eval_forward", &batchnorm_eval_forward,
          "Custom BatchNorm eval forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));

    m.def("bmm_forward", &bmm_forward,
          "Custom batch matmul forward",
          py::arg("mat1"), py::arg("mat2"));

    m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
          "Custom ConvTranspose2d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("batch"), py::arg("in_channels"), py::arg("out_channels"),
          py::arg("kernel_h"), py::arg("kernel_w"), py::arg("stride_h"), py::arg("stride_w"),
          py::arg("padding_h"), py::arg("padding_w"),
          py::arg("output_padding_h"), py::arg("output_padding_w"),
          py::arg("seq_len"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
