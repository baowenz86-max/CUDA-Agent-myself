// Binding for cat_dim1 and clone_tensor CUDA kernels
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void cat_dim1_launcher(
    const float* a, const float* b, float* out,
    int N, int C1, int C2, int H, int W,
    long long a_stride_n, long long a_stride_c, long long a_stride_h, long long a_stride_w,
    long long b_stride_n, long long b_stride_c, long long b_stride_h, long long b_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void clone_tensor_launcher(const float* src, float* dst, long long numel, cudaStream_t stream);

// Implementation of cat_dim1 operator
torch::Tensor cat_dim1(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.is_cuda(), "cat_dim1 input a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "cat_dim1 input b must be a CUDA tensor");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "cat_dim1 input a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "cat_dim1 input b must be float32");
    TORCH_CHECK(a.dim() == 4 && b.dim() == 4, "cat_dim1 expects 4D tensors");
    TORCH_CHECK(a.size(0) == b.size(0), "batch dimension mismatch");
    TORCH_CHECK(a.size(2) == b.size(2) && a.size(3) == b.size(3), "spatial dimensions mismatch");

    int N = a.size(0);
    int C1 = a.size(1);
    int C2 = b.size(1);
    int H = a.size(2);
    int W = a.size(3);

    // Allocate output contiguous
    torch::Tensor out = torch::empty({N, C1 + C2, H, W}, a.options());

    // Compute strides (assuming contiguous output)
    long long out_stride_w = 1;
    long long out_stride_h = W;
    long long out_stride_c = (long long)H * W;
    long long out_stride_n = (long long)(C1 + C2) * H * W;

    // Get input strides from tensors (may be non-contiguous)
    long long a_stride_n = a.stride(0);
    long long a_stride_c = a.stride(1);
    long long a_stride_h = a.stride(2);
    long long a_stride_w = a.stride(3);
    long long b_stride_n = b.stride(0);
    long long b_stride_c = b.stride(1);
    long long b_stride_h = b.stride(2);
    long long b_stride_w = b.stride(3);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    cat_dim1_launcher(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        N, C1, C2, H, W,
        a_stride_n, a_stride_c, a_stride_h, a_stride_w,
        b_stride_n, b_stride_c, b_stride_h, b_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    // Check for CUDA errors after launch (optional but good practice)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "cat_dim1 CUDA error: ", cudaGetErrorString(err));

    return out;
}

// Implementation of clone_tensor operator
torch::Tensor clone_tensor(torch::Tensor x) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "clone_tensor input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "clone_tensor input must be float32");

    // Allocate output like input (contiguous)
    torch::Tensor out = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    clone_tensor_launcher(
        x.data_ptr<float>(), out.data_ptr<float>(), x.numel(), stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "clone_tensor CUDA error: ", cudaGetErrorString(err));

    return out;
}

// Registration function
void register_custom_ops_binding(pybind11::module& m) {
    m.def("cat_dim1", &cat_dim1, "Concatenate two tensors along dimension 1 (channel)",
          py::arg("a"), py::arg("b"));
    m.def("clone_tensor", &clone_tensor, "Clone a tensor (copy)",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
