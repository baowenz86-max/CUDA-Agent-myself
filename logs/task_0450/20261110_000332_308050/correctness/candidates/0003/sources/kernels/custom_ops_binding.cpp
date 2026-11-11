#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void embedding_lookup_launcher(const int64_t* input,
                                           const float* weight,
                                           float* output,
                                           int B,
                                           int L,
                                           int E,
                                           cudaStream_t stream);

extern "C" void linear_launcher(const float* input,
                                const float* weight,
                                const float* bias,
                                float* output,
                                int M,
                                int K,
                                int N,
                                cudaStream_t stream);

extern "C" void cross_entropy_loss_launcher(const float* input,
                                            const int64_t* target,
                                            float* loss,
                                            int B,
                                            int L,
                                            int C,
                                            cudaStream_t stream);

// ------------------- Embedding Lookup -------------------
torch::Tensor embedding_lookup_forward(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "Embedding lookup input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Embedding lookup weight must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Embedding lookup input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Embedding lookup weight must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kInt64, "Embedding lookup input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Embedding lookup weight must be float32");
    TORCH_CHECK(input.dim() == 2, "Embedding lookup input must be 2D [B, L]");
    TORCH_CHECK(weight.dim() == 2, "Embedding lookup weight must be 2D [num_embeddings, embedding_dim]");

    int64_t B = input.size(0);
    int64_t L = input.size(1);
    int64_t E = weight.size(1);

    // output shape [B, L, E]
    auto output = torch::empty({B, L, E}, weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_lookup_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(L),
        static_cast<int>(E),
        stream
    );

    return output;
}

// ------------------- Linear -------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(input.is_cuda(), "Linear input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Linear weight must be CUDA tensor");
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "Linear bias must be CUDA tensor");
    }
    TORCH_CHECK(input.is_contiguous(), "Linear input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "Linear weight must be contiguous");
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_contiguous(), "Linear bias must be contiguous");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Linear input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Linear weight must be float32");
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "Linear bias must be float32");
    }
    TORCH_CHECK(input.dim() == 3, "Linear input must be 3D [B, L, in_features]");
    TORCH_CHECK(weight.dim() == 2, "Linear weight must be 2D [out_features, in_features]");

    int64_t B = input.size(0);
    int64_t L = input.size(1);
    int64_t K = input.size(2);
    int64_t N = weight.size(0);
    TORCH_CHECK(weight.size(1) == K, "Linear weight in_features must match input feature dim");
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().dim() == 1 && bias.value().size(0) == N, "Linear bias size must match out_features");
    }

    int64_t M = B * L;
    // output shape [B, L, out_features]
    auto output = torch::empty({B, L, N}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        bias_ptr = bias.value().data_ptr<float>();
    }

    linear_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        static_cast<int>(M),
        static_cast<int>(K),
        static_cast<int>(N),
        stream
    );

    return output;
}

// ------------------- Cross Entropy Loss -------------------
torch::Tensor cross_entropy_loss_forward(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "Cross entropy input must be CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "Cross entropy target must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Cross entropy input must be contiguous");
    TORCH_CHECK(target.is_contiguous(), "Cross entropy target must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Cross entropy input must be float32");
    TORCH_CHECK(target.dtype() == torch::kInt64, "Cross entropy target must be int64");
    TORCH_CHECK(input.dim() == 3, "Cross entropy input must be 3D [B, L, C]");
    TORCH_CHECK(target.dim() == 2, "Cross entropy target must be 2D [B, L]");

    int64_t B = input.size(0);
    int64_t L = input.size(1);
    int64_t C = input.size(2);
    TORCH_CHECK(target.size(0) == B && target.size(1) == L, "Cross entropy target shape must match input batch and length");

    // scalar loss tensor
    auto loss = torch::empty({}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    cross_entropy_loss_launcher(
        input.data_ptr<float>(),
        target.data_ptr<int64_t>(),
        loss.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(L),
        static_cast<int>(C),
        stream
    );

    return loss;
}

// ------------------- Registration -------------------
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_lookup_forward", &embedding_lookup_forward,
          "Embedding lookup forward",
          py::arg("input"), py::arg("weight"));
    m.def("linear_forward", &linear_forward,
          "Linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("cross_entropy_loss_forward", &cross_entropy_loss_forward,
          "Cross entropy loss forward",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
