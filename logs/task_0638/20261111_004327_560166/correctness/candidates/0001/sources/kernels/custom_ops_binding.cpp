#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding1_forward_launcher(
    float* output,
    const int64_t* input,
    const float* weight,
    int batch,
    int seq_len,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream
);

extern "C" void embedding2_forward_launcher(
    float* output,
    const int64_t* input,
    const float* weight,
    int batch,
    int seq_len,
    int embedding_dim,
    int num_embeddings,
    cudaStream_t stream
);

extern "C" void add_tensors_launcher(
    float* output,
    const float* a,
    const float* b,
    int total,
    cudaStream_t stream
);

static torch::Tensor embedding_forward_common(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    int num_embeddings,
    const std::string& op_name
) {
    TORCH_CHECK(input.is_cuda(), op_name, ": input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), op_name, ": weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt64, op_name, ": input must be int64");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, op_name, ": weight must be float32");
    TORCH_CHECK(input.is_contiguous(), op_name, ": input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), op_name, ": weight must be contiguous");
    TORCH_CHECK(input.dim() == 2, op_name, ": input must be 2D [B, L]");
    TORCH_CHECK(weight.dim() == 2, op_name, ": weight must be 2D [E, D]");

    int batch = input.size(0);
    int seq_len = input.size(1);
    int embedding_dim = weight.size(1);
    TORCH_CHECK(weight.size(0) == num_embeddings, op_name, ": weight first dim must equal num_embeddings");

    auto output = torch::empty({batch, seq_len, embedding_dim}, weight.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (op_name == "embedding1_forward") {
        embedding1_forward_launcher(
            output.data_ptr<float>(),
            input.data_ptr<int64_t>(),
            weight.data_ptr<float>(),
            batch, seq_len, embedding_dim, num_embeddings,
            stream
        );
    } else { // embedding2_forward
        embedding2_forward_launcher(
            output.data_ptr<float>(),
            input.data_ptr<int64_t>(),
            weight.data_ptr<float>(),
            batch, seq_len, embedding_dim, num_embeddings,
            stream
        );
    }

    return output;
}

torch::Tensor embedding1_forward(torch::Tensor input, torch::Tensor weight) {
    return embedding_forward_common(input, weight, 1024, "embedding1_forward");
}

torch::Tensor embedding2_forward(torch::Tensor input, torch::Tensor weight) {
    return embedding_forward_common(input, weight, 512, "embedding2_forward");
}

torch::Tensor add_tensors(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "add_tensors: a must be CUDA");
    TORCH_CHECK(b.is_cuda(), "add_tensors: b must be CUDA");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "add_tensors: a must be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "add_tensors: b must be float32");
    TORCH_CHECK(a.is_contiguous(), "add_tensors: a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "add_tensors: b must be contiguous");
    TORCH_CHECK(a.sizes() == b.sizes(), "add_tensors: input shapes must match");

    auto output = torch::empty_like(a);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_tensors_launcher(
        output.data_ptr<float>(),
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        a.numel(),
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding1_forward", &embedding1_forward, "Embedding lookup for embedding1",
          py::arg("input"), py::arg("weight"));
    m.def("embedding2_forward", &embedding2_forward, "Embedding lookup for embedding2",
          py::arg("input"), py::arg("weight"));
    m.def("add_tensors", &add_tensors, "Elementwise addition",
          py::arg("a"), py::arg("b"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
