#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
    void tile_launcher(const float* input, float* output, int64_t N, int64_t C, int64_t L, int64_t R, cudaStream_t stream);
    int conv1d_launcher(const float* x, const float* weight, const float* bias, float* y,
                        int N, int C_in, int C_out, int kernel_size, int L,
                        int stride, int pad, int dilation, int groups, cudaStream_t stream);
}

// tile_forward: repeat batch dimension
// x: [N, C, L] -> [N*repeats, C, L]
torch::Tensor tile_forward(torch::Tensor x, int64_t repeats) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (N, C, L)");
    TORCH_CHECK(repeats > 0, "repeats must be positive");

    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);

    auto output = torch::empty({N * repeats, C, L}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tile_launcher(x.data_ptr<float>(), output.data_ptr<float>(), N, C, L, repeats, stream);

    return output;
}

// conv1d_forward: Conv1d with fixed stride=1, padding=1, dilation=1, groups=1
// x: [N, C_in, L]
// weight: [C_out, C_in, kernel_size]
// bias: [C_out] (may be None)
// output: [N, C_out, out_L]
torch::Tensor conv1d_forward(torch::Tensor x, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (N, C, L)");

    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.is_contiguous(), "Weight must be contiguous");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (C_out, C_in, kernel_size)");

    int64_t N = x.size(0);
    int64_t C_in = x.size(1);
    int64_t L = x.size(2);
    int64_t C_out = weight.size(0);
    int64_t kernel_size = weight.size(2);
    TORCH_CHECK(weight.size(1) == C_in, "Weight in_channels must match input in_channels");

    torch::Tensor bias_tensor;
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.is_contiguous(), "Bias must be contiguous");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias_tensor.dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias_tensor.size(0) == C_out, "Bias size must match out_channels");
        bias_ptr = bias_tensor.data_ptr<float>();
    }

    int64_t out_L = (L + 2 * 1 - 1 * (kernel_size - 1) - 1) / 1 + 1;
    auto output = torch::empty({N, C_out, out_L}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int status = conv1d_launcher(
        x.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(kernel_size),
        static_cast<int>(L),
        1, // stride
        1, // padding
        1, // dilation
        1, // groups
        stream
    );
    TORCH_CHECK(status == 0, "conv1d_forward launcher failed with code ", status);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("tile_forward", &tile_forward,
          "Tile the batch dimension (repeat)",
          py::arg("x"), py::arg("repeats"));
    m.def("conv1d_forward", &conv1d_forward,
          "Conv1d forward using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);