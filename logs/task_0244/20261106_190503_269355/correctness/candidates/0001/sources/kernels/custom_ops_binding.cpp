// PyTorch binding for custom tanh and batch_norm2d evaluation kernels.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Launchers exported from custom_ops.cu
extern "C" void tanh_forward_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    long long in_stride_n,  long long in_stride_c,  long long in_stride_h,  long long in_stride_w,
    cudaStream_t stream);

extern "C" void batch_norm2d_forward_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    long long in_stride_n,  long long in_stride_c,  long long in_stride_h,  long long in_stride_w,
    cudaStream_t stream);

// Helper: get strides as long long array from tensor, assuming 4D
static inline std::array<long long, 4> get_strides_ll(const torch::Tensor& t) {
    TORCH_CHECK(t.dim() == 4, "Expected 4D tensor");
    return {t.stride(0), t.stride(1), t.stride(2), t.stride(3)};
}

// tanh_forward: elementwise tanh, output has same strides as input
torch::Tensor tanh_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    auto out = torch::empty_strided(x.sizes(), x.strides(), x.options());

    auto in_strides = get_strides_ll(x);
    auto out_strides = get_strides_ll(out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    tanh_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        x.size(0), x.size(1), x.size(2), x.size(3),
        out_strides[0], out_strides[1], out_strides[2], out_strides[3],
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        stream);

    return out;
}

// batch_norm2d_forward: evaluation-mode batch norm, output strides match input
torch::Tensor batch_norm2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D");

    for (const auto& t : {weight, bias, running_mean, running_var}) {
        TORCH_CHECK(t.is_cuda(), "Parameter must be a CUDA tensor");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "Parameter must be float32");
        TORCH_CHECK(t.is_contiguous(), "Parameter must be contiguous");
        TORCH_CHECK(t.numel() == x.size(1), "Parameter size must equal channels");
    }

    auto out = torch::empty_strided(x.sizes(), x.strides(), x.options());

    auto in_strides = get_strides_ll(x);
    auto out_strides = get_strides_ll(out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm2d_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        x.size(0), x.size(1), x.size(2), x.size(3),
        out_strides[0], out_strides[1], out_strides[2], out_strides[3],
        in_strides[0], in_strides[1], in_strides[2], in_strides[3],
        stream);

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("tanh_forward", &tanh_forward, "Elementwise tanh");
    m.def("batch_norm2d_forward", &batch_norm2d_forward, "BatchNorm2d evaluation",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("running_mean"), pybind11::arg("running_var"), pybind11::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops)
