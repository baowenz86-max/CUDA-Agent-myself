#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_softmax_launcher(
    float* output, const float* input,
    int B, int C, int L,
    int in_stride_b, int in_stride_c, int in_stride_l,
    int out_stride_b, int out_stride_c, int out_stride_l,
    int dim,
    cudaStream_t stream);

extern "C" void custom_cumprod_launcher(
    float* output, const float* input,
    int B, int C, int L,
    int in_stride_b, int in_stride_c, int in_stride_l,
    int out_stride_b, int out_stride_c, int out_stride_l,
    int dim,
    cudaStream_t stream);

// Helper to validate tensor and extract dimensions
void validate_input(const torch::Tensor& x, int64_t& B, int64_t& C, int64_t& L) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D [B, C, L]");
    B = x.size(0);
    C = x.size(1);
    L = x.size(2);
}

// custom_softmax forward
at::Tensor custom_softmax(at::Tensor x, int64_t dim = 1) {
    int64_t B, C, L;
    validate_input(x, B, C, L);
    TORCH_CHECK(dim == 1, "custom_softmax currently only supports dim=1");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_softmax_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(x.stride(0)),
        static_cast<int>(x.stride(1)),
        static_cast<int>(x.stride(2)),
        static_cast<int>(output.stride(0)),
        static_cast<int>(output.stride(1)),
        static_cast<int>(output.stride(2)),
        static_cast<int>(dim),
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// custom_cumprod forward
at::Tensor custom_cumprod(at::Tensor x, int64_t dim = 2) {
    int64_t B, C, L;
    validate_input(x, B, C, L);
    TORCH_CHECK(dim == 2, "custom_cumprod currently only supports dim=2");

    auto output = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_cumprod_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        static_cast<int>(x.stride(0)),
        static_cast<int>(x.stride(1)),
        static_cast<int>(x.stride(2)),
        static_cast<int>(output.stride(0)),
        static_cast<int>(output.stride(1)),
        static_cast<int>(output.stride(2)),
        static_cast<int>(dim),
        stream
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_softmax", &custom_softmax, "Custom softmax along dim",
          py::arg("x"), py::arg("dim") = 1);
    m.def("custom_cumprod", &custom_cumprod, "Custom cumprod along dim",
          py::arg("x"), py::arg("dim") = 2);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
