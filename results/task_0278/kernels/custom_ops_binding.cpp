#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_elu_forward_launcher(
    const float* input,
    float* output,
    long long total_elements,
    float alpha,
    cudaStream_t stream);

extern "C" void custom_max_pool2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    cudaStream_t stream);

// PyTorch wrapper for custom_elu_forward
torch::Tensor custom_elu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_elu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_elu_forward: input must be float32");

    // Make contiguous if necessary; the launcher assumes contiguous memory.
    auto x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_elu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<long long>(x_contig.numel()),
        1.0f,  // alpha = 1.0 as in nn.ELU()
        stream
    );

    return output;
}

// PyTorch wrapper for custom_max_pool2d_forward
torch::Tensor custom_max_pool2d_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_max_pool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "custom_max_pool2d_forward: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_max_pool2d_forward: input must be 4D (N, C, H, W)");

    // Pooling parameters: kernel_size=(2,2), stride=(2,2), padding=0, dilation=1, ceil_mode=false
    const int pool_kernel_h = 2;
    const int pool_kernel_w = 2;
    const int stride_h = 2;
    const int stride_w = 2;

    auto x_contig = x.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);

    int H_out = (H - pool_kernel_h) / stride_h + 1;
    int W_out = (W - pool_kernel_w) / stride_w + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "custom_max_pool2d_forward: output size must be positive");

    auto output = torch::empty({N, C, H_out, W_out}, x_contig.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_max_pool2d_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_elu_forward", &custom_elu_forward, "Custom ELU forward (CUDA)",
          py::arg("x"));
    m.def("custom_max_pool2d_forward", &custom_max_pool2d_forward, "Custom MaxPool2d forward (CUDA)",
          py::arg("x"));
}

// Auto-register
REGISTER_BINDING(custom_ops, register_custom_ops);
