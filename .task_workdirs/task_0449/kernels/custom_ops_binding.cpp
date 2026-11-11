#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void lrn_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int size, float alpha, float beta, float k,
    cudaStream_t stream);

extern "C" void lrn_forward_strided_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    int size, float alpha, float beta, float k,
    int stride_n, int stride_c, int stride_h, int stride_w,
    cudaStream_t stream);

// PyTorch wrapper for LRN forward
torch::Tensor lrn_forward(torch::Tensor x, int64_t size, double alpha, double beta, double k) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input tensor must be float32");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4-dimensional (N, C, H, W)");
    TORCH_CHECK(size > 0, "size must be positive");

    auto output = torch::empty_like(x);

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Determine if contiguous; if not, use strided launcher with actual strides
    if (x.is_contiguous()) {
        lrn_forward_launcher(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            N, C, H, W,
            static_cast<int>(size),
            static_cast<float>(alpha),
            static_cast<float>(beta),
            static_cast<float>(k),
            stream);
    } else {
        int stride_n = x.stride(0);
        int stride_c = x.stride(1);
        int stride_h = x.stride(2);
        int stride_w = x.stride(3);
        lrn_forward_strided_launcher(
            output.data_ptr<float>(),
            x.data_ptr<float>(),
            N, C, H, W,
            static_cast<int>(size),
            static_cast<float>(alpha),
            static_cast<float>(beta),
            static_cast<float>(k),
            stride_n, stride_c, stride_h, stride_w,
            stream);
    }

    return output;
}

// Registration
void register_lrn_forward(pybind11::module& m) {
    m.def("lrn_forward", &lrn_forward,
          "Local Response Normalization forward",
          py::arg("x"), py::arg("size"), py::arg("alpha"), py::arg("beta"), py::arg("k"));
}

REGISTER_BINDING(lrn_forward, register_lrn_forward);
