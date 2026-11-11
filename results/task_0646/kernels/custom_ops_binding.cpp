#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sum_dim1_launcher(
    const float* x,
    float* out,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t sx_N,
    int64_t sx_C,
    int64_t sx_H,
    int64_t sx_W,
    int64_t total_output,
    cudaStream_t stream);

extern "C" void log_softmax_dim0_launcher(
    const float* x,
    float* out,
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t total_spatial,
    cudaStream_t stream);

// PyTorch binding for sum over dimension 1
torch::Tensor sum_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sum_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "sum_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_dim1: input must be 4D");
    TORCH_CHECK(x.size(1) == 3, "sum_dim1: dimension 1 must have size 3");

    // Ensure input is contiguous for the launcher's assumed linear addressing
    auto x_contig = x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t H = x_contig.size(2);
    int64_t W = x_contig.size(3);
    int64_t total_output = N * H * W;

    // Output is contiguous [N, H, W]
    auto out = torch::empty({N, H, W}, x_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher with contiguous strides
    sum_dim1_launcher(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        H,
        W,
        C * H * W, // stride N
        H * W,     // stride C
        W,         // stride H
        1,         // stride W
        total_output,
        stream
    );

    return out;
}

// PyTorch binding for log_softmax over dimension 0
torch::Tensor log_softmax_dim0(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "log_softmax_dim0: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "log_softmax_dim0: input must be float32");
    TORCH_CHECK(x.dim() == 3, "log_softmax_dim0: input must be 3D");

    // Ensure contiguous for the launcher's linear addressing
    auto x_contig = x.contiguous();

    int64_t N = x_contig.size(0);
    int64_t H = x_contig.size(1);
    int64_t W = x_contig.size(2);
    int64_t total_spatial = H * W;

    // Output same shape as input
    auto out = torch::empty_like(x_contig);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    log_softmax_dim0_launcher(
        x_contig.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        H,
        W,
        total_spatial,
        stream
    );

    return out;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("sum_dim1", &sum_dim1, "Reduce over dimension 1 (channels)", py::arg("x"));
    m.def("log_softmax_dim0", &log_softmax_dim0, "Log softmax over dimension 0", py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
