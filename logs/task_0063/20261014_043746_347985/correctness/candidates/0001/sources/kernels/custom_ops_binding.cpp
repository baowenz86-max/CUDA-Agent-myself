#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_avgpool1d_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L_in,
    int L_out,
    cudaStream_t stream);

extern "C" void unfold_launcher(
    const float* input,
    float* output,
    int N,
    int C,
    int L,
    int kernel_size,
    int stride,
    int padding,
    int L_out,
    cudaStream_t stream);

// Wrapper for adaptive_avgpool1d_forward
torch::Tensor adaptive_avgpool1d_forward(torch::Tensor x, int64_t output_size) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must have shape [N, C, L]");
    TORCH_CHECK(output_size > 0, "output_size must be positive");

    // Ensure contiguous (input from avgpool is contiguous, but be safe)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int N = x.size(0);
    int C = x.size(1);
    int L_in = x.size(2);
    int L_out = static_cast<int>(output_size);

    auto output = torch::empty({N, C, L_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    adaptive_avgpool1d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        L_in,
        L_out,
        stream
    );

    return output;
}

// Wrapper for unfold_forward
torch::Tensor unfold_forward(torch::Tensor x, int64_t kernel_size, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must have shape [N, C, L]");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    // Ensure contiguous
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }

    int N = x.size(0);
    int C = x.size(1);
    int L = x.size(2);
    int k = static_cast<int>(kernel_size);
    int s = static_cast<int>(stride);
    int p = static_cast<int>(padding);
    int L_out = (L + 2 * p - k) / s + 1;
    TORCH_CHECK(L_out > 0, "Invalid unfold parameters: L_out must be positive");

    auto output = torch::empty({N, C * k, L_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    unfold_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N,
        C,
        L,
        k,
        s,
        p,
        L_out,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("adaptive_avgpool1d_forward", &adaptive_avgpool1d_forward,
          "Adaptive average pooling 1D forward",
          pybind11::arg("x"), pybind11::arg("output_size"));
    m.def("unfold_forward", &unfold_forward,
          "Unfold (im2col) forward",
          pybind11::arg("x"), pybind11::arg("kernel_size"),
          pybind11::arg("stride"), pybind11::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
