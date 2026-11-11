#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void custom_max_dim1_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int64_t sN, int64_t sC, int64_t sH, int64_t sW,
    cudaStream_t stream);

extern "C" void custom_argmax_dim1_launcher(
    const float* input,
    int64_t* output,
    int N, int C, int H, int W,
    int64_t sN, int64_t sC, int64_t sH, int64_t sW,
    cudaStream_t stream);

extern "C" void custom_scalar_mul_launcher(
    const float* input,
    float* output,
    float scalar_value,
    int total,
    cudaStream_t stream);

// Helper to check CUDA tensor properties
static void check_cuda_float_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.dim() == 4, name, " must be 4-dimensional");
}

// PyTorch wrapper for custom_max_dim1
torch::Tensor custom_max_dim1(torch::Tensor x) {
    check_cuda_float_tensor(x, "x");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output tensor [N, 1, H, W]
    auto output = torch::empty({N, 1, H, W}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Obtain raw pointers and strides (int64_t)
    const float* input_ptr = x.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();
    int64_t sN = x.stride(0);
    int64_t sC = x.stride(1);
    int64_t sH = x.stride(2);
    int64_t sW = x.stride(3);

    // Call launcher (launcher internally handles non-contiguous input by making a contiguous copy if needed)
    custom_max_dim1_launcher(input_ptr, output_ptr,
                             static_cast<int>(N), static_cast<int>(C),
                             static_cast<int>(H), static_cast<int>(W),
                             sN, sC, sH, sW,
                             stream);

    // Check for errors after launch
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_max_dim1 launch failed");

    return output;
}

// PyTorch wrapper for custom_argmax_dim1
torch::Tensor custom_argmax_dim1(torch::Tensor x) {
    check_cuda_float_tensor(x, "x");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Allocate output tensor [N, 1, H, W] of int64
    auto output = torch::empty({N, 1, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* input_ptr = x.data_ptr<float>();
    int64_t* output_ptr = output.data_ptr<int64_t>();
    int64_t sN = x.stride(0);
    int64_t sC = x.stride(1);
    int64_t sH = x.stride(2);
    int64_t sW = x.stride(3);

    custom_argmax_dim1_launcher(input_ptr, output_ptr,
                                static_cast<int>(N), static_cast<int>(C),
                                static_cast<int>(H), static_cast<int>(W),
                                sN, sC, sH, sW,
                                stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_argmax_dim1 launch failed");

    return output;
}

// PyTorch wrapper for custom_scalar_mul
torch::Tensor custom_scalar_mul(torch::Tensor input, torch::Tensor scalar) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4-dimensional");
    TORCH_CHECK(scalar.dtype() == torch::kFloat32, "scalar must be float32");
    TORCH_CHECK(scalar.numel() == 1, "scalar must have exactly one element");

    int64_t total = input.numel();

    // Allocate output with same shape as input
    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* input_ptr = input.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Retrieve scalar value from tensor; works for both CPU and CUDA, synchronizes if necessary
    float scalar_value = scalar.item<float>();

    custom_scalar_mul_launcher(input_ptr, output_ptr, scalar_value,
                               static_cast<int>(total), stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "custom_scalar_mul launch failed");

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim1", &custom_max_dim1, "Custom max along dim=1 (keepdim)",
          py::arg("x"));
    m.def("custom_argmax_dim1", &custom_argmax_dim1, "Custom argmax along dim=1 (keepdim)",
          py::arg("x"));
    m.def("custom_scalar_mul", &custom_scalar_mul, "Multiply tensor by scalar",
          py::arg("input"), py::arg("scalar"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
