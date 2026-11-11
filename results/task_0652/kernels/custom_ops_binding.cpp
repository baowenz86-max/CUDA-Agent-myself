#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sum_dim_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t in_stride_b,
    int64_t in_stride_c,
    int64_t in_stride_h,
    int64_t in_stride_w,
    int64_t out_stride_b,
    int64_t out_stride_c,
    int64_t out_stride_h,
    int64_t out_stride_w,
    cudaStream_t stream
);

extern "C" void leaky_relu_launcher(
    float* output,
    const float* input,
    int64_t numel,
    float negative_slope,
    cudaStream_t stream
);

// Wrapper for sum over dimension 1 with keepdim=True
static torch::Tensor sum_dim_forward(torch::Tensor input, int64_t dim, bool keepdim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (B, C, H, W)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(keepdim, "keepdim must be true for this op");

    // Ensure input is contiguous for straightforward stride computation
    auto input_contig = input.contiguous();

    int64_t B = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    // Output shape [B, 1, H, W]
    auto output = torch::empty({B, 1, H, W}, input_contig.options());

    // Strides for contiguous tensors (assuming row-major)
    int64_t in_stride_b = C * H * W;
    int64_t in_stride_c = H * W;
    int64_t in_stride_h = W;
    int64_t in_stride_w = 1;
    int64_t out_stride_b = H * W;
    int64_t out_stride_c = H * W; // not used since output has C=1
    int64_t out_stride_h = W;
    int64_t out_stride_w = 1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        B, C, H, W,
        in_stride_b, in_stride_c, in_stride_h, in_stride_w,
        out_stride_b, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

// Wrapper for LeakyReLU element-wise
static torch::Tensor leaky_relu_forward(torch::Tensor input, double negative_slope) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");

    // Ensure contiguous for element-wise kernel
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int64_t numel = input_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        numel,
        static_cast<float>(negative_slope),
        stream
    );

    return output;
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("sum_dim", &sum_dim_forward,
          "Sum over dimension 1 with keepdim=True",
          py::arg("input"), py::arg("dim"), py::arg("keepdim"));
    m.def("leaky_relu", &leaky_relu_forward,
          "LeakyReLU element-wise",
          py::arg("input"), py::arg("negative_slope"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
