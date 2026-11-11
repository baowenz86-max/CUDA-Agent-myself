#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void max_dim1_keepdim_launcher(
    const float* x, float* values, int64_t* indices,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    long long so_b, long long so_c, long long so_h, long long so_w,
    cudaStream_t stream);

extern "C" void max_dim2_no_keepdim_launcher(
    const float* x, float* values, int64_t* indices,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    long long so_b, long long so_c, long long so_w,
    cudaStream_t stream);

// Helper to check CUDA errors after kernel launch
static void check_cuda_error(cudaError_t err, const char* op_name) {
    TORCH_CHECK(err == cudaSuccess, op_name, " CUDA error: ", cudaGetErrorString(err));
}

// Wrapper for max over dim=1 with keepdim=True
// Input: [B, C, H, W] float32 (supports strides)
// Output: torch.return_types.max with values [B,1,H,W] float32 and indices [B,1,H,W] int64
torch::return_types::max max_dim1_keepdim(torch::Tensor x) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "max_dim1_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "max_dim1_keepdim: input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "max_dim1_keepdim: input must be 4D [B, C, H, W]");
    int B = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(B > 0 && C > 0 && H > 0 && W > 0, "max_dim1_keepdim: dimensions must be positive");

    // Output tensors: values [B,1,H,W] float32, indices [B,1,H,W] int64
    auto values = torch::empty({B, 1, H, W}, x.options());
    auto indices = torch::empty({B, 1, H, W}, x.options().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    max_dim1_keepdim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        stream
    );
    check_cuda_error(cudaGetLastError(), "max_dim1_keepdim");

    // Construct torch.return_types.max
    torch::return_types::max result;
    result.values = values;
    result.indices = indices;
    return result;
}

// Wrapper for max over dim=2 with keepdim=False
// Input: [B, 1, H, W] float32 (supports strides)
// Output: torch.return_types.max with values [B,1,W] float32 and indices [B,1,W] int64
torch::return_types::max max_dim2_no_keepdim(torch::Tensor x) {
    // Input checks
    TORCH_CHECK(x.is_cuda(), "max_dim2_no_keepdim: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "max_dim2_no_keepdim: input dtype must be float32");
    TORCH_CHECK(x.dim() == 4, "max_dim2_no_keepdim: input must be 4D [B, 1, H, W]");
    TORCH_CHECK(x.size(1) == 1, "max_dim2_no_keepdim: dimension 1 must be 1");
    int B = x.size(0);
    int C = 1; // always 1
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(B > 0 && H > 0 && W > 0, "max_dim2_no_keepdim: dimensions must be positive");

    // Output tensors: values [B,1,W] float32, indices [B,1,W] int64
    auto values = torch::empty({B, 1, W}, x.options());
    auto indices = torch::empty({B, 1, W}, x.options().dtype(torch::kInt64));

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Launch kernel
    max_dim2_no_keepdim_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        B, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        values.stride(0), values.stride(1), values.stride(2),
        stream
    );
    check_cuda_error(cudaGetLastError(), "max_dim2_no_keepdim");

    // Construct torch.return_types.max
    torch::return_types::max result;
    result.values = values;
    result.indices = indices;
    return result;
}

// Register bindings
void register_custom_ops_binding(pybind11::module& m) {
    m.def("max_dim1_keepdim", &max_dim1_keepdim, "Max over dim=1 with keepdim=True", py::arg("x"));
    m.def("max_dim2_no_keepdim", &max_dim2_no_keepdim, "Max over dim=2 with keepdim=False", py::arg("x"));
}

REGISTER_BINDING(custom_ops_binding, register_custom_ops_binding);
