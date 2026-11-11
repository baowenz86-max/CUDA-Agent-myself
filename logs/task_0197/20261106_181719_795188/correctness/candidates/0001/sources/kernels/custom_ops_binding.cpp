#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void avgpool2d_forward_launcher(const float* input,
                                           float* output,
                                           int N, int C, int H, int W,
                                           int kernel_size, int stride, int padding,
                                           cudaStream_t stream);

extern "C" void relu_forward_launcher(const float* input,
                                      float* output,
                                      int total,
                                      cudaStream_t stream);

// Wrapper for AvgPool2d
torch::Tensor avgpool2d_forward(torch::Tensor input, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must have shape [N, C, H, W]");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Compute output spatial dimensions
    // Matches torch.nn.functional.avg_pool2d
    int H_out = (H + 2 * static_cast<int>(padding) - static_cast<int>(kernel_size)) / static_cast<int>(stride) + 1;
    int W_out = (W + 2 * static_cast<int>(padding) - static_cast<int>(kernel_size)) / static_cast<int>(stride) + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "kernel_size/stride/padding invalid for input size");

    auto output = torch::empty({N, C, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    avgpool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for ReLU
torch::Tensor relu_forward(torch::Tensor input, bool inplace) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    // For this task, inplace is always False; but we respect parameter.
    // If inplace is true, we would modify input and return input. However,
    // contract only exercises inplace=False. We still implement it correctly
    // to avoid undefined behavior if someone passes True.
    if (inplace) {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        relu_forward_launcher(
            input.data_ptr<float>(),
            input.data_ptr<float>(),
            input.numel(),
            stream
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        return input;
    } else {
        auto output = torch::empty_like(input);
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
        relu_forward_launcher(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.numel(),
            stream
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
        return output;
    }
}

void register_custom_ops(pybind11::module& m) {
    m.def("avgpool2d_forward", &avgpool2d_forward,
          "AvgPool2d forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding") = 0);

    m.def("relu_forward", &relu_forward,
          "ReLU forward (CUDA)",
          pybind11::arg("input"),
          pybind11::arg("inplace") = false);
}

REGISTER_BINDING(custom_ops, register_custom_ops);