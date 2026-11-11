#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <tuple>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_max_dim1_launcher(
    const float* x,
    float* values,
    int64_t* indices,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream
);

extern "C" void custom_roll_dim2_launcher(
    const float* x,
    float* y,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream
);

// PyTorch wrapper for max over dim=1 keepdim=True
std::tuple<torch::Tensor, torch::Tensor> custom_max_dim1_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // The kernel assumes contiguous row-major layout for this specific shape.
    // Ensure x is contiguous for correct indexing.
    auto x_contig = x.contiguous();
    const float* x_ptr = x_contig.data_ptr<float>();

    auto values = torch::empty({N, 1, H, W}, x.options());
    auto indices = torch::empty({N, 1, H, W}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_max_dim1_launcher(
        x_ptr,
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        N, C, H, W,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_max_dim1 kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(values, indices);
}

// PyTorch wrapper for roll along dim=2 with shift=1
torch::Tensor custom_roll_dim2_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N, C, H, W)");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    // Ensure contiguous
    auto x_contig = x.contiguous();
    const float* x_ptr = x_contig.data_ptr<float>();

    auto y = torch::empty_like(x_contig);
    float* y_ptr = y.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_roll_dim2_launcher(
        x_ptr,
        y_ptr,
        N, C, H, W,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_roll_dim2 kernel launch failed: ", cudaGetErrorString(err));

    return y;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_max_dim1", &custom_max_dim1_forward,
          "Max over dim=1 with keepdim=True, returns (values, indices)",
          py::arg("x"));
    m.def("custom_roll_dim2", &custom_roll_dim2_forward,
          "Roll along dim=2 by 1 (cyclic shift)",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
