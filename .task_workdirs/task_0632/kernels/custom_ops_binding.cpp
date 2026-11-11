#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declarations from custom_ops.cu
extern "C" {
void interpolate_bilinear_launcher(
    const float* input,
    float* output,
    int64_t in_batch, int64_t in_channels, int64_t in_height, int64_t in_width,
    int64_t out_height, int64_t out_width,
    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_b, int64_t out_stride_c, int64_t out_stride_h, int64_t out_stride_w,
    double scale_factor,
    cudaStream_t stream);

void sum_reduction_launcher(
    const float* input,
    float* output,
    int64_t batch, int64_t channels, int64_t height, int64_t width,
    int64_t in_stride_b, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    int64_t out_stride_b, int64_t out_stride_c,
    cudaStream_t stream);
}

// Interpolate bilinear wrapper
torch::Tensor interpolate_bilinear(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (batch, channels, height, width)");

    // Make contiguous if needed
    auto x_contig = x.contiguous();

    int64_t in_batch = x_contig.size(0);
    int64_t in_channels = x_contig.size(1);
    int64_t in_height = x_contig.size(2);
    int64_t in_width = x_contig.size(3);

    int64_t out_height = static_cast<int64_t>(std::floor(in_height * scale_factor));
    int64_t out_width = static_cast<int64_t>(std::floor(in_width * scale_factor));

    auto output = torch::empty({in_batch, in_channels, out_height, out_width}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    interpolate_bilinear_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        in_batch, in_channels, in_height, in_width,
        out_height, out_width,
        x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        scale_factor,
        stream);

    return output;
}

// Sum reduction wrapper accepting an int dim
// For this task, dim must be 2, meaning reduce over dimensions 2 and 3.
torch::Tensor sum_reduction(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_reduction currently supports 4D input");
    TORCH_CHECK(dim == 2, "Only reduction over dim=2 (last two dims) is supported");

    auto x_contig = x.contiguous();

    int64_t batch = x_contig.size(0);
    int64_t channels = x_contig.size(1);
    int64_t height = x_contig.size(2);
    int64_t width = x_contig.size(3);

    auto output = torch::empty({batch, channels}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_reduction_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, height, width,
        x_contig.stride(0), x_contig.stride(1), x_contig.stride(2), x_contig.stride(3),
        output.stride(0), output.stride(1),
        stream);

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("interpolate_bilinear", &interpolate_bilinear,
          "Bilinear interpolation (align_corners=False)",
          py::arg("x"), py::arg("scale_factor"));
    m.def("sum_reduction", &sum_reduction,
          "Sum over specified dim (only dim=2 supported)",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
