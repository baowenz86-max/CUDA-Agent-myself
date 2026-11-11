#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations from custom_ops.cu
extern "C" void custom_hardsigmoid_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream);

extern "C" void custom_instance_norm2d_no_affine_launcher(
    const float* input,
    float* output,
    float eps,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t stride_n, int64_t stride_c, int64_t stride_h, int64_t stride_w,
    cudaStream_t stream);

static void validate_cuda_float_tensor_4d(const torch::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D [N,C,H,W]");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must have dtype torch.float32");
}

torch::Tensor custom_hardsigmoid(torch::Tensor x) {
    validate_cuda_float_tensor_4d(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t total_elements = x.numel();

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_hardsigmoid_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        N, C, H, W,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_hardsigmoid kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor custom_instance_norm2d_no_affine(torch::Tensor x, double eps) {
    validate_cuda_float_tensor_4d(x);

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_instance_norm2d_no_affine_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_instance_norm2d_no_affine kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_hardsigmoid", &custom_hardsigmoid,
          "Custom hard sigmoid: y = clamp(x+3, 0, 6) / 6",
          pybind11::arg("x"));
    m.def("custom_instance_norm2d_no_affine", &custom_instance_norm2d_no_affine,
          "InstanceNorm2d without affine: (x - mean) / sqrt(var + eps)",
          pybind11::arg("x"), pybind11::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
