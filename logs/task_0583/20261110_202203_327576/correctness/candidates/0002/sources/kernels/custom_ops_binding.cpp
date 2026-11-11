#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void isinf_like_launcher(
    const float* input,
    bool* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void where_inf_replace_launcher(
    const bool* mask,
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream);

extern "C" void logsoftmax_dim1_launcher(
    const float* input,
    float* output,
    int64_t N,
    int64_t C,
    int64_t H,
    int64_t W,
    int64_t strideN,
    int64_t strideC,
    int64_t strideH,
    int64_t strideW,
    cudaStream_t stream);

// Helper to check CUDA tensor and dtype
void check_cuda_float(torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

void check_cuda_bool(torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kBool, name, " must be bool");
}

// isinf_like wrapper
torch::Tensor isinf_like(torch::Tensor x) {
    check_cuda_float(x, "x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto output = torch::empty(x.sizes(), x.options().dtype(torch::kBool));
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    isinf_like_launcher(
        x.data_ptr<float>(),
        output.data_ptr<bool>(),
        numel,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "isinf_like kernel failed: ", cudaGetErrorString(err));
    return output;
}

// where_inf_replace wrapper
torch::Tensor where_inf_replace(torch::Tensor inf_mask, torch::Tensor x) {
    check_cuda_bool(inf_mask, "inf_mask");
    check_cuda_float(x, "x");
    TORCH_CHECK(inf_mask.sizes() == x.sizes(), "inf_mask and x must have the same shape");
    TORCH_CHECK(inf_mask.is_contiguous(), "inf_mask must be contiguous");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    auto output = torch::empty_like(x);
    int64_t numel = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    where_inf_replace_launcher(
        inf_mask.data_ptr<bool>(),
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "where_inf_replace kernel failed: ", cudaGetErrorString(err));
    return output;
}

// logsoftmax_dim1 wrapper
torch::Tensor logsoftmax_dim1(torch::Tensor x) {
    check_cuda_float(x, "x");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    // Require contiguous? The launcher handles arbitrary strides; we just need
    // to pass the strides to the kernel. We allocate output with same strides as input.
    // The kernel uses input strides to index both input and output, so output must have
    // identical strides for correct memory layout. We allocate with empty_strided.
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t strideN = x.stride(0);
    int64_t strideC = x.stride(1);
    int64_t strideH = x.stride(2);
    int64_t strideW = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    logsoftmax_dim1_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        strideN, strideC, strideH, strideW,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "logsoftmax_dim1 kernel failed: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("isinf_like", &isinf_like, "Elementwise isinf detection",
          py::arg("x"));
    m.def("where_inf_replace", &where_inf_replace, "Replace inf with -inf based on mask",
          py::arg("inf_mask"), py::arg("x"));
    m.def("logsoftmax_dim1", &logsoftmax_dim1, "Log softmax along dim=1",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
