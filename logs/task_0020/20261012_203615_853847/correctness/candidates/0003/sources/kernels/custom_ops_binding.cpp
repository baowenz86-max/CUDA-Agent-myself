#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv_transpose1d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch,
    int in_channels,
    int out_channels,
    int in_length,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream
);

extern "C" void sign_forward(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream
);

extern "C" void embedding_forward(
    const long long* indices,
    const float* weight,
    float* output,
    int batch,
    int length,
    int embedding_dim,
    cudaStream_t stream
);

// Wrapper for conv_transpose1d
static torch::Tensor conv_transpose1d_forward_tensor(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_length = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "bias out_channels mismatch");

    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "computed output length must be positive");

    auto output = torch::empty({batch, out_channels, out_length}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        out_channels,
        in_length,
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    return output;
}

// Wrapper for sign
static torch::Tensor sign_forward_tensor(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sign_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.numel(),
        stream
    );
    return output;
}

// Wrapper for embedding
static torch::Tensor embedding_forward_tensor(
    torch::Tensor indices,
    torch::Tensor weight
) {
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    TORCH_CHECK(indices.dim() == 2, "indices must be 2D (batch, length)");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (num_embeddings, embedding_dim)");

    int batch = indices.size(0);
    int length = indices.size(1);
    int embedding_dim = weight.size(1);

    auto output = torch::empty({batch, length, embedding_dim}, weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    embedding_forward(
        indices.data_ptr<long long>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        length,
        embedding_dim,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward_tensor,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("sign_forward", &sign_forward_tensor,
          "Sign forward",
          py::arg("input"));
    m.def("embedding_forward", &embedding_forward_tensor,
          "Embedding forward",
          py::arg("indices"), py::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
