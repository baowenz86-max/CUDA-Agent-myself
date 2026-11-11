// Bindings for custom_leaky_relu and custom_avg_pool2d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void custom_leaky_relu_launcher(
    void* out,
    const void* in,
    float negative_slope,
    int64_t total_elements,
    cudaStream_t stream
);

extern "C" void custom_avg_pool2d_launcher(
    void* out,
    const void* in,
    int64_t N,
    int64_t C,
    int64_t H_in,
    int64_t W_in,
    int64_t H_out,
    int64_t W_out,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int64_t stride_n,
    int64_t stride_c,
    int64_t stride_h_in,
    int64_t stride_w_in,
    int64_t stride_n_out,
    int64_t stride_c_out,
    int64_t stride_h_out,
    int64_t stride_w_out,
    cudaStream_t stream
);

torch::Tensor custom_leaky_relu(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "custom_leaky_relu: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_leaky_relu: input must be float32");

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_leaky_relu_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<float>(negative_slope),
        x.numel(),
        stream
    );

    return out;
}

torch::Tensor custom_avg_pool2d(
    torch::Tensor x,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding
) {
    TORCH_CHECK(x.is_cuda(), "custom_avg_pool2d: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_avg_pool2d: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_avg_pool2d: input must be 4D (N, C, H, W)");

    // Ensure input is contiguous for efficient access (but launcher supports strides anyway)
    auto x_contig = x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t H_in = x_contig.size(2);
    int64_t W_in = x_contig.size(3);

    int64_t H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    int64_t W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "custom_avg_pool2d: invalid output dimensions");

    auto out = torch::empty({N, C, H_out, W_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_avg_pool2d_launcher(
        out.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        N, C, H_in, W_in, H_out, W_out,
        static_cast<int>(kernel_size), static_cast<int>(kernel_size),
        static_cast<int>(stride), static_cast<int>(stride),
        static_cast<int>(padding), static_cast<int>(padding),
        x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        stream
    );

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_leaky_relu", &custom_leaky_relu,
          "LeakyReLU forward (CUDA)",
          pybind11::arg("x"), pybind11::arg("negative_slope"));
    m.def("custom_avg_pool2d", &custom_avg_pool2d,
          "AvgPool2d forward (CUDA)",
          pybind11::arg("x"), pybind11::arg("kernel_size"),
          pybind11::arg("stride"), pybind11::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
