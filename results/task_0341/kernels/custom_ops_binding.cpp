#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int B, int H, int E,
    cudaStream_t stream);

extern "C" void avg_pool1d_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int L,
    int kernel_size, int stride, int padding,
    int outL,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_l,
    cudaStream_t stream);

// embedding_forward: input int64 [B,H], weight float32 [V,E] -> output float32 [B,H,E]
torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [B,H]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [V,E]");

    int B = input.size(0);
    int H = input.size(1);
    int E = weight.size(1);
    int V = weight.size(0);
    TORCH_CHECK(V > 0, "weight rows must be >0");

    auto output = torch::empty({B, H, E}, input.options().dtype(torch::kFloat32));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, E,
        stream);
    return output;
}

// transpose_forward: return a view with swapped dim0 and dim1 (no computation)
torch::Tensor transpose_forward(torch::Tensor input, int64_t dim0, int64_t dim1) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() > 0, "input must have positive rank");
    int64_t ndim = input.dim();
    // Normalize negative dims
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    TORCH_CHECK(dim0 >= 0 && dim0 < ndim, "dim0 out of range");
    TORCH_CHECK(dim1 >= 0 && dim1 < ndim, "dim1 out of range");
    // PyTorch transpose returns a view, no data movement
    return input.transpose(dim0, dim1);
}

// avg_pool1d_forward: input [B,C,L] (possibly non-contiguous), kernel_size/stride/padding ints
torch::Tensor avg_pool1d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [B,C,L]");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int B = input.size(0);
    int C = input.size(1);
    int L = input.size(2);
    int K = static_cast<int>(kernel_size);
    int S = static_cast<int>(stride);
    int P = static_cast<int>(padding);

    // Compute output length as PyTorch does: floor((L + 2*P - K) / S) + 1
    int outL = (L + 2*P - K) / S + 1;
    TORCH_CHECK(outL > 0, "output length must be positive");

    // Allocate contiguous output [B,C,outL]
    auto output = torch::empty({B, C, outL}, input.options());

    // Get strides for input in terms of elements (long long to avoid overflow)
    auto strides = input.strides();
    long long in_stride_b = strides[0];
    long long in_stride_c = strides[1];
    long long in_stride_l = strides[2];

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avg_pool1d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, L,
        K, S, P,
        outL,
        in_stride_b, in_stride_c, in_stride_l,
        stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward,
          "Embedding lookup forward",
          py::arg("input"), py::arg("weight"));
    m.def("transpose_forward", &transpose_forward,
          "Transpose dimensions (view)",
          py::arg("input"), py::arg("dim0"), py::arg("dim1"));
    m.def("avg_pool1d_forward", &avg_pool1d_forward,
          "1D average pooling forward",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
