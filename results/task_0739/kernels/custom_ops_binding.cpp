#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avg_pool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    int64_t out_h, int64_t out_w,
    cudaStream_t stream);

extern "C" void softmax_dim1_forward_launcher(
    const float* input,
    float* output,
    int64_t N, int64_t C, int64_t H, int64_t W,
    cudaStream_t stream);

// PyTorch wrapper for adaptive_avg_pool2d_forward
torch::Tensor adaptive_avg_pool2d_forward(torch::Tensor input, int64_t out_h, int64_t out_w) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    TORCH_CHECK(out_h > 0 && out_w > 0, "out_h and out_w must be positive");

    auto output = torch::empty({N, C, out_h, out_w}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    adaptive_avg_pool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        out_h, out_w,
        stream
    );

    return output;
}

// PyTorch wrapper for softmax_dim1_forward
torch::Tensor softmax_dim1_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 4, "Input must be 4D (N,C,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_dim1_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avg_pool2d_forward", &adaptive_avg_pool2d_forward,
          "Adaptive average pooling forward",
          py::arg("input"), py::arg("out_h"), py::arg("out_w"));
    m.def("softmax_dim1_forward", &softmax_dim1_forward,
          "Softmax along channel dimension",
          py::arg("input"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
