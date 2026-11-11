// kernels/custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/return_types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void erf_forward_launcher(
    const float* input, float* output,
    int64_t rows, int64_t cols,
    int64_t in_stride0, int64_t in_stride1,
    int64_t out_stride0, int64_t out_stride1,
    cudaStream_t stream);

extern "C" void median_dim0_forward_launcher(
    const float* input, float* values_out, int64_t* indices_out,
    int rows, int cols,
    int64_t in_stride0, int64_t in_stride1,
    cudaStream_t stream);

// Elementwise erf forward
static torch::Tensor erf_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");

    auto output = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    const int64_t rows = x.size(0);
    const int64_t cols = x.size(1);
    const int64_t in_stride0 = x.stride(0);
    const int64_t in_stride1 = x.stride(1);
    const int64_t out_stride0 = output.stride(0);
    const int64_t out_stride1 = output.stride(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    erf_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        rows, cols,
        in_stride0, in_stride1,
        out_stride0, out_stride1,
        stream
    );

    return output;
}

// Median along dim 0, returns a torch.return_types.median container
static auto median_dim0_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be 2D");

    const int rows = static_cast<int>(x.size(0));
    const int cols = static_cast<int>(x.size(1));
    const int64_t in_stride0 = x.stride(0);
    const int64_t in_stride1 = x.stride(1);

    auto values = torch::empty({cols}, x.options().dtype(torch::kFloat32));
    auto indices = torch::empty({cols}, x.options().dtype(torch::kInt64));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    median_dim0_forward_launcher(
        x.data_ptr<float>(),
        values.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        rows, cols,
        in_stride0, in_stride1,
        stream
    );

    return torch::return_types::median(values, indices);
}

// Registration function
static void register_custom_ops(pybind11::module& m) {
    m.def("erf_forward", &erf_forward, "Elementwise erf forward",
          py::arg("x"));
    m.def("median_dim0_forward", &median_dim0_forward, "Median along dimension 0",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
