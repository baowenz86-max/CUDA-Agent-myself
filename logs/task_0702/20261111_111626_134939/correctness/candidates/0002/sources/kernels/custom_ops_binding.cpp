#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void sum_dim1_launcher(
    const float* x,
    float* out,
    int N, int C, int H, int W,
    long long sxN, long long sxC, long long sxH, long long sxW,
    long long soN, long long soH, long long soW,
    cudaStream_t stream);

extern "C" void log_softmax_dim0_launcher(
    const float* x,
    float* out,
    int N, long long inner_size,
    long long sxN, long long sxInner,
    long long soN, long long soInner,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// sum_dim1 wrapper
// -----------------------------------------------------------------------------
torch::Tensor sum_dim1(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "sum_dim1: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "sum_dim1: input must be float32");
    TORCH_CHECK(x.dim() == 4, "sum_dim1: input must be 4D [N,C,H,W]");

    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    TORCH_CHECK(C > 0, "sum_dim1: C dimension must be positive");

    // Output shape [N, H, W]
    auto out = torch::empty({N, H, W}, x.options());

    // Get strides in elements (PyTorch strides are already in elements)
    auto x_strides = x.strides();
    auto out_strides = out.strides();

    long long sxN = x_strides[0];
    long long sxC = x_strides[1];
    long long sxH = x_strides[2];
    long long sxW = x_strides[3];

    long long soN = out_strides[0];
    long long soH = out_strides[1];
    long long soW = out_strides[2];

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sum_dim1_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, C, H, W,
        sxN, sxC, sxH, sxW,
        soN, soH, soW,
        stream);

    return out;
}

// -----------------------------------------------------------------------------
// log_softmax_dim0 wrapper
// -----------------------------------------------------------------------------
torch::Tensor log_softmax_dim0(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "log_softmax_dim0: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "log_softmax_dim0: input must be float32");
    TORCH_CHECK(x.dim() >= 1, "log_softmax_dim0: input must have at least 1 dimension");

    int N = x.size(0);
    TORCH_CHECK(N > 0, "log_softmax_dim0: N dimension must be positive");

    // inner_size = product of remaining dimensions
    long long inner_size = 1;
    for (int i = 1; i < x.dim(); ++i) {
        inner_size *= x.size(i);
    }

    TORCH_CHECK(inner_size > 0, "log_softmax_dim0: inner_size must be positive");

    // Output same shape
    auto out = torch::empty_like(x);

    // Treat input as 2D [N, inner_size] with strides
    long long sxN = x.stride(0);
    // For dim=1, stride of inner dimension is 1; for higher dims, the stride of the fastest varying dimension
    // is the correct effective stride for the flattened inner dimension.
    long long sxInner = (x.dim() == 1) ? 1 : x.stride(x.dim() - 1);

    // For output, we just need same shape and strides; output is contiguous with soN = inner_size, soInner = 1
    long long soN = out.stride(0);
    long long soInner = (out.dim() == 1) ? 1 : out.stride(out.dim() - 1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_dim0_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        N, inner_size,
        sxN, sxInner,
        soN, soInner,
        stream);

    return out;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("sum_dim1", &sum_dim1, "sum over dim=1", py::arg("x"));
    m.def("log_softmax_dim0", &log_softmax_dim0, "log_softmax over dim=0", py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
