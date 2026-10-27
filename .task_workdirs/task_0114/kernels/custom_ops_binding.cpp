#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void elu_forward_launcher(const float* input, float* output, int64_t numel, float alpha, cudaStream_t stream);
extern "C" void digamma_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void triu_forward_launcher(const float* input, float* output,
                                    int64_t rows, int64_t cols,
                                    int64_t in_stride_row, int64_t in_stride_col,
                                    int64_t out_stride_row, int64_t out_stride_col,
                                    cudaStream_t stream);

// ELU wrapper
torch::Tensor elu_forward(torch::Tensor x, double alpha = 1.0) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
    // Contract says support strides, no need contiguous. But our launcher assumes contiguous? The ELU kernel uses flat index, works with any contiguous storage.
    // However, if input is non-contiguous with strides, flat indexing won't map correctly. The reference input is contiguous [2048,4096] with stride [4096,1].
    // We can accept any strides? Our kernel uses flat index, but if x is not contiguous, data_ptr points to contiguous memory? Actually non-contiguous tensor has storage but with strides, data_ptr points to the start but indexing in kernel should use strides.
    // To be safe and align with contract 'supports strides', we should make a contiguous copy if not contiguous.
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    elu_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), static_cast<float>(alpha), stream);
    return output;
}

// Digamma wrapper
torch::Tensor digamma_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor cannot be empty");
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    digamma_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), x.numel(), stream);
    return output;
}

// Triu wrapper
torch::Tensor triu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 2, "Input must be a 2D tensor");
    TORCH_CHECK(x.size(0) > 0 && x.size(1) > 0, "Input tensor cannot be empty");
    // Ensure contiguous for stride simplicity; but contract says supports strides, we can handle strides in launcher.
    // We'll pass actual strides and copy output with same shape and strides? Output should be like x (same shape, dtype, device). Contract says shape same as x, strides not specified but likely same as input if contiguous.
    // We'll create output with same sizes and strides as x (empty_like preserves strides). But our launcher uses passed strides, so it can handle any strides.
    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int64_t rows = x.size(0);
    int64_t cols = x.size(1);
    int64_t in_stride_row = x.stride(0);
    int64_t in_stride_col = x.stride(1);
    int64_t out_stride_row = output.stride(0);
    int64_t out_stride_col = output.stride(1);
    triu_forward_launcher(x.data_ptr<float>(), output.data_ptr<float>(), rows, cols,
                          in_stride_row, in_stride_col, out_stride_row, out_stride_col, stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("elu_forward", &elu_forward, "ELU forward",
          py::arg("x"), py::arg("alpha") = 1.0);
    m.def("digamma_forward", &digamma_forward, "Digamma forward",
          py::arg("x"));
    m.def("triu_forward", &triu_forward, "Triu forward",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
