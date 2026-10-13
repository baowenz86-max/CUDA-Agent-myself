#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void upsample_bilinear2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    float scale_factor,
    cudaStream_t stream);

extern "C" void logical_and_range_launcher(
    const float* input,
    bool* output,
    int total,
    cudaStream_t stream);

static cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

torch::Tensor upsample_bilinear2d(torch::Tensor x, double scale_factor = 2.0) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int outH = static_cast<int>(H * scale_factor);
    int outW = static_cast<int>(W * scale_factor);

    auto output = torch::empty({N, C, outH, outW}, x.options());

    cudaStream_t stream = getCurrentCUDAStream();
    upsample_bilinear2d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, outH, outW,
        static_cast<float>(scale_factor),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in upsample_bilinear2d: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor logical_and_range(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    int total = x.numel();
    auto output = torch::empty(x.sizes(), x.options().dtype(torch::kBool));

    cudaStream_t stream = getCurrentCUDAStream();
    logical_and_range_launcher(
        x.data_ptr<float>(),
        output.data_ptr<bool>(),
        total,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in logical_and_range: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("upsample_bilinear2d", &upsample_bilinear2d,
          py::arg("x"), py::arg("scale_factor") = 2.0);
    m.def("logical_and_range", &logical_and_range,
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
