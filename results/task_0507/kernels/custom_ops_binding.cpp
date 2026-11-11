#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    cudaStream_t stream
);

extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int kernel_size, int stride,
    cudaStream_t stream
);

static torch::Tensor relu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "relu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "relu_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "relu_forward: input must be contiguous");

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    relu_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        stream
    );
    return output;
}

static torch::Tensor avgpool2d_forward(torch::Tensor x, int64_t kernel_size, int64_t stride) {
    TORCH_CHECK(x.is_cuda(), "avgpool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "avgpool2d_forward: input must be float32");
    TORCH_CHECK(x.is_contiguous(), "avgpool2d_forward: input must be contiguous");
    TORCH_CHECK(x.dim() == 4, "avgpool2d_forward: input must be 4D NCHW");
    TORCH_CHECK(kernel_size > 0 && stride > 0, "avgpool2d_forward: kernel_size and stride must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t outH = (H - kernel_size) / stride + 1;
    int64_t outW = (W - kernel_size) / stride + 1;
    TORCH_CHECK(outH > 0 && outW > 0, "avgpool2d_forward: output size must be positive");

    auto output = torch::empty({N, C, outH, outW}, x.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avgpool2d_forward_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        stream
    );
    return output;
}

static void register_custom_ops(pybind11::module& m) {
    m.def("relu_forward", &relu_forward, "ReLU forward",
          py::arg("x"));
    m.def("avgpool2d_forward", &avgpool2d_forward, "AvgPool2d forward",
          py::arg("x"), py::arg("kernel_size"), py::arg("stride"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
