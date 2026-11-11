#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declarations of CUDA launchers from custom_ops.cu
extern "C" void custom_tril_launcher(const float* input, float* output,
                                    int B, int C, int H, int W,
                                    long long total_elements,
                                    cudaStream_t stream);

extern "C" void custom_elu_launcher(const float* input, float* output,
                                    long long total_elements,
                                    cudaStream_t stream);

extern "C" void custom_add_bias_launcher(const float* input, const float* bias,
                                         float* output,
                                         int B, int C, int H, int W,
                                         long long total_elements,
                                         cudaStream_t stream);

// ------------------------- custom_tril binding -------------------------
torch::Tensor custom_tril(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_tril: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_tril: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_tril: input must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_tril: input must be 4D");

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const long long total_elements = x.numel();

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_tril_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        total_elements,
        stream
    );

    return output;
}

// ------------------------- custom_elu binding -------------------------
torch::Tensor custom_elu(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_elu: input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_elu: input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_elu: input must be float32");

    const long long total_elements = x.numel();

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_elu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        stream
    );

    return output;
}

// ------------------------- custom_add_bias binding -------------------------
torch::Tensor custom_add_bias(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "custom_add_bias: x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "custom_add_bias: bias must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "custom_add_bias: x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "custom_add_bias: bias must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_add_bias: x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "custom_add_bias: bias must be float32");
    TORCH_CHECK(x.dim() == 4, "custom_add_bias: x must be 4D");
    TORCH_CHECK(bias.dim() == 3, "custom_add_bias: bias must be 3D");
    TORCH_CHECK(bias.size(0) == x.size(1), "custom_add_bias: bias first dim must match channel count");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "custom_add_bias: bias spatial dims must be 1x1");

    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const long long total_elements = x.numel();

    auto output = torch::empty_like(x);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_add_bias_launcher(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, H, W,
        total_elements,
        stream
    );

    return output;
}

// ------------------------- Registration -------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("custom_tril", &custom_tril, "Apply torch.tril to a 4D contiguous float32 tensor", py::arg("x"));
    m.def("custom_elu", &custom_elu, "Apply ELU (alpha=1.0) elementwise", py::arg("x"));
    m.def("custom_add_bias", &custom_add_bias, "Add broadcasted bias of shape [C,1,1]",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
