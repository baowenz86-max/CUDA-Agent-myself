#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int64_t batch_size,
    int64_t seq_length,
    int64_t embedding_dim,
    cudaStream_t stream);

extern "C" void mean_dim_forward_launcher(
    const float* input,
    float* output,
    int64_t batch_size,
    int64_t seq_length,
    int64_t embedding_dim,
    int64_t dim,
    cudaStream_t stream);

extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    cudaStream_t stream);

extern "C" void bce_with_logits_forward_launcher(
    const float* logits,
    const float* target,
    float* output,
    int64_t batch_size,
    cudaStream_t stream);

// Wrapper for embedding_forward
torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, seq_length]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [num_embeddings, embedding_dim]");

    int64_t batch_size = input.size(0);
    int64_t seq_length = input.size(1);
    int64_t embedding_dim = weight.size(1);

    auto output = torch::empty({batch_size, seq_length, embedding_dim}, weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_length,
        embedding_dim,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for mean_dim_forward
torch::Tensor mean_dim_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch_size, seq_length, embedding_dim]");
    TORCH_CHECK(dim == 1, "only dim=1 is supported");

    int64_t batch_size = input.size(0);
    int64_t seq_length = input.size(1);
    int64_t embedding_dim = input.size(2);

    auto output = torch::empty({batch_size, embedding_dim}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mean_dim_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_length,
        embedding_dim,
        dim,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "mean_dim_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for linear_forward
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [batch_size, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");

    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias->dim() == 1, "bias must be 1D [out_features]");
        TORCH_CHECK(bias->size(0) == weight.size(0), "bias size must match out_features");
    }

    int64_t batch_size = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = weight.size(0);

    auto output = torch::empty({batch_size, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.has_value() ? bias->data_ptr<float>() : nullptr;

    linear_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "linear_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Wrapper for bce_with_logits_forward
torch::Tensor bce_with_logits_forward(torch::Tensor logits, torch::Tensor target) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "logits must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch_size, 1]");
    TORCH_CHECK(target.dim() == 2, "target must be 2D [batch_size, 1]");
    TORCH_CHECK(logits.size(0) == target.size(0), "batch_size mismatch");
    TORCH_CHECK(logits.size(1) == 1 && target.size(1) == 1, "second dimension must be 1");

    int64_t batch_size = logits.size(0);

    auto output = torch::empty({}, logits.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bce_with_logits_forward_launcher(
        logits.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bce_with_logits_forward kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward",
          py::arg("input"), py::arg("weight"));
    m.def("mean_dim_forward", &mean_dim_forward, "Mean along dim",
          py::arg("input"), py::arg("dim"));
    m.def("linear_forward", &linear_forward, "Linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("bce_with_logits_forward", &bce_with_logits_forward, "BCE with logits forward",
          py::arg("logits"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
