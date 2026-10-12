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
    int batch_size,
    int in_channels,
    int out_channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    int output_padding,
    cudaStream_t stream
);

extern "C" void sign_forward(
    const float* input,
    float* output,
    int n,
    cudaStream_t stream
);

extern "C" void embedding_forward(
    const int64_t* indices,
    const float* weight,
    float* output,
    int num_indices,
    int embedding_dim,
    cudaStream_t stream
);

// ConvTranspose1d binding
at::Tensor conv_transpose1d(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int64_t kernel_size,
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
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(bias.dtype() == at::kFloat, "bias must be float32");

    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t length_in = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t k = weight.size(2);

    TORCH_CHECK(kernel_size == k, "kernel_size mismatch");
    TORCH_CHECK(weight.size(0) == in_channels, "in_channels mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "out_channels mismatch");

    int64_t length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(length_out > 0, "computed output length must be positive");

    auto output = at::empty({batch_size, out_channels, length_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(in_channels),
        static_cast<int>(out_channels),
        static_cast<int>(length_in),
        static_cast<int>(length_out),
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(output_padding),
        stream
    );

    return output;
}

// Sign binding
at::Tensor sign(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");

    auto output = at::empty_like(input);
    int numel = static_cast<int>(input.numel());
    TORCH_CHECK(numel <= INT_MAX, "input too large for int indexing");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sign_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        stream
    );

    return output;
}

// Embedding binding
at::Tensor embedding(at::Tensor indices, at::Tensor weight) {
    TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D (num_embeddings, embedding_dim)");
    TORCH_CHECK(indices.dtype() == at::kLong, "indices must be int64");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");

    int64_t num_indices = indices.numel();
    int64_t embedding_dim = weight.size(1);

    auto output = at::empty({num_indices, embedding_dim}, weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    embedding_forward(
        indices.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(num_indices),
        static_cast<int>(embedding_dim),
        stream
    );

    return output;
}

// Register all functions
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("kernel_size"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("sign_forward", &sign, "Sign forward", py::arg("input"));
    m.def("embedding_forward", &embedding, "Embedding forward", py::arg("indices"), py::arg("weight"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
