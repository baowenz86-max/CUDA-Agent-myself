#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_upsample_bilinear_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    double scale_factor,
    int64_t outH, int64_t outW,
    int64_t in_stride_n, int64_t in_stride_c, int64_t in_stride_h, int64_t in_stride_w,
    cudaStream_t stream);

extern "C" void custom_softmax_dim_launcher(
    const float* input,
    float* output,
    const int64_t* shape,
    int64_t ndim,
    int64_t dim,
    const int64_t* stride,
    int64_t total_numel,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// custom_upsample_bilinear wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_upsample_bilinear(torch::Tensor x, double scale_factor) {
    TORCH_CHECK(x.is_cuda(), "custom_upsample_bilinear: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_upsample_bilinear: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_upsample_bilinear: input must be 4D (N,C,H,W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    int64_t outH = static_cast<int64_t>(std::floor(H * scale_factor));
    int64_t outW = static_cast<int64_t>(std::floor(W * scale_factor));

    TORCH_CHECK(outH > 0 && outW > 0, "custom_upsample_bilinear: scale_factor too small");

    // Allocate contiguous output
    auto output = torch::empty({N, C, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_upsample_bilinear_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        scale_factor,
        outH, outW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// custom_softmax_dim wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_softmax_dim(torch::Tensor x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "custom_softmax_dim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_softmax_dim: input must be float32");

    int64_t ndim = x.dim();
    TORCH_CHECK(ndim > 0, "custom_softmax_dim: input must have at least one dimension");

    // Normalize negative dim
    if (dim < 0) dim += ndim;
    TORCH_CHECK(dim >= 0 && dim < ndim, "custom_softmax_dim: dim out of range");

    // Allocate contiguous output with same shape
    auto output = torch::empty_like(x);

    // Prepare shape and stride arrays on host (launcher copies them to device?)
    // The launcher signature expects pointers to host or device memory.
    // In this binding we pass host pointers; the current launcher only reads them
    // on host before kernel launch, so host memory is fine.
    std::vector<int64_t> shape_vec(ndim);
    std::vector<int64_t> stride_vec(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        shape_vec[i] = x.size(i);
        stride_vec[i] = x.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_dim_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        shape_vec.data(),
        ndim,
        dim,
        stride_vec.data(),
        x.numel(),
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_upsample_bilinear", &custom_upsample_bilinear,
          "Bilinear upsampling with scale_factor (align_corners=False)",
          py::arg("x"), py::arg("scale_factor"));
    m.def("custom_softmax_dim", &custom_softmax_dim,
          "Softmax along given dimension",
          py::arg("x"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
