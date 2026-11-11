#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void reduce_mean_dim1_launcher(
    const float* x,
    float* y,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream);

extern "C" void mul_scalar_launcher(
    const float* x,
    float* y,
    float scalar,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_x_n, int64_t stride_x_c, int64_t stride_x_h, int64_t stride_x_w,
    int64_t stride_y_n, int64_t stride_y_c, int64_t stride_y_h, int64_t stride_y_w,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// reduce_mean_dim1 forward
// -----------------------------------------------------------------------------
torch::Tensor reduce_mean_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "reduce_mean_dim1 input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "reduce_mean_dim1 input must be float32");
    TORCH_CHECK(x.dim() == 4, "reduce_mean_dim1 input must be 4D (N,C,H,W)");

    auto sizes = x.sizes();
    int64_t N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];

    // Allocate output with shape [N,1,H,W], same device and dtype
    std::vector<int64_t> out_sizes = {N, 1, H, W};
    auto y = torch::empty(out_sizes, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Strides (assuming contiguous or non-contiguous, rely on tensor.stride)
    int64_t sx_n = x.stride(0), sx_c = x.stride(1), sx_h = x.stride(2), sx_w = x.stride(3);
    int64_t sy_n = y.stride(0), sy_c = y.stride(1), sy_h = y.stride(2), sy_w = y.stride(3);

    reduce_mean_dim1_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        sx_n, sx_c, sx_h, sx_w,
        sy_n, sy_c, sy_h, sy_w,
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// mul_scalar forward
// -----------------------------------------------------------------------------
torch::Tensor mul_scalar(torch::Tensor x, double scalar) {
    TORCH_CHECK(x.is_cuda(), "mul_scalar input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "mul_scalar input must be float32");
    TORCH_CHECK(x.dim() == 4, "mul_scalar input must be 4D (N,C,H,W)");

    auto sizes = x.sizes();
    int64_t N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];

    auto y = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    int64_t sx_n = x.stride(0), sx_c = x.stride(1), sx_h = x.stride(2), sx_w = x.stride(3);
    int64_t sy_n = y.stride(0), sy_c = y.stride(1), sy_h = y.stride(2), sy_w = y.stride(3);

    mul_scalar_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        static_cast<float>(scalar),
        N, C, H, W,
        sx_n, sx_c, sx_h, sx_w,
        sy_n, sy_c, sy_h, sy_w,
        stream);

    return y;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("reduce_mean_dim1", &reduce_mean_dim1, "Reduce mean over dimension 1 (keepdim=True)",
          pybind11::arg("x"));
    m.def("mul_scalar", &mul_scalar, "Multiply tensor by scalar",
          pybind11::arg("x"), pybind11::arg("scalar"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
