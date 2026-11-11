#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#include "../binding_registry.h"

// Extern declarations from custom_ops.cu
extern "C" void hardsigmoid_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    long long stride_n,
    long long stride_c,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream);

extern "C" void mean_last_2_dims_forward_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    long long stride_n,
    long long stride_c,
    long long stride_h,
    long long stride_w,
    cudaStream_t stream);

// Helper to check CUDA tensor properties
static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must have dtype float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4-dimensional");
}

// hardsigmoid_forward: element-wise hardsigmoid, output with same strides as input
torch::Tensor hardsigmoid_forward(torch::Tensor x) {
    check_cuda_float32(x, "x");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Ensure dimensions fit in int
    TORCH_CHECK(N >= 0 && C >= 0 && H >= 0 && W >= 0, "Input dimensions must be non-negative");
    TORCH_CHECK((long long)N * C * H * W <= (long long)INT_MAX, "Total number of elements exceeds int range");

    // Allocate output with same sizes and strides as input
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    // Get strides in elements (PyTorch strides are in elements, not bytes)
    long long stride_n = x.stride(0);
    long long stride_c = x.stride(1);
    long long stride_h = x.stride(2);
    long long stride_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    hardsigmoid_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        stream);

    return output;
}

// mean_last_2_dims_forward: reduce over last two dimensions -> [N, C]
torch::Tensor mean_last_2_dims_forward(torch::Tensor x) {
    check_cuda_float32(x, "x");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    TORCH_CHECK(N >= 0 && C >= 0 && H >= 0 && W >= 0, "Input dimensions must be non-negative");
    TORCH_CHECK(H > 0 && W > 0, "H and W must be positive for mean over last two dims");

    // Output is [N, C]
    auto output = torch::empty({N, C}, x.options());

    long long stride_n = x.stride(0);
    long long stride_c = x.stride(1);
    long long stride_h = x.stride(2);
    long long stride_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_last_2_dims_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stride_n, stride_c, stride_h, stride_w,
        stream);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("hardsigmoid_forward", &hardsigmoid_forward,
          "x: torch.Tensor -> torch.Tensor",
          pybind11::arg("x"));
    m.def("mean_last_2_dims_forward", &mean_last_2_dims_forward,
          "x: torch.Tensor -> torch.Tensor",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
