#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void embedding_forward_launcher(
    const int64_t* input_gpu,
    const float* weight_gpu,
    float* output_gpu,
    int total_rows,
    int embedding_dim,
    cudaStream_t stream);

extern "C" void log_softmax_forward_launcher(
    const float* input_gpu,
    float* output_gpu,
    int total_rows,
    int D,
    cudaStream_t stream);

extern "C" void softmax_forward_launcher(
    const float* input_gpu,
    float* output_gpu,
    int total_rows,
    int D,
    cudaStream_t stream);

extern "C" void kl_div_batchmean_forward_launcher_v2(
    const float* input_gpu,
    const float* target_gpu,
    float* output_gpu,
    int batch_size,
    int total_elements,
    cudaStream_t stream);

static void check_cuda_float(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

static void check_cuda_int64(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kInt64, name, " must be int64");
}

// embedding_forward: input CUDA int64 [B,S], weight CUDA float32 [V,D] -> output CUDA float32 [B,S,D]
// If input or weight is on CPU, it will be moved to CUDA first.
torch::Tensor embedding_forward(torch::Tensor input, torch::Tensor weight) {
    // Ensure tensors are on CUDA (matching execution device)
    if (!input.is_cuda()) {
        input = input.cuda();
    }
    if (!weight.is_cuda()) {
        weight = weight.cuda();
    }

    TORCH_CHECK(input.dtype() == torch::kInt64, "input must be int64");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");

    int64_t batch_size = input.size(0);
    int64_t sequence_length = input.size(1);
    int64_t num_embeddings = weight.size(0);
    int64_t embedding_dim = weight.size(1);
    int64_t total_rows = batch_size * sequence_length;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::empty({batch_size, sequence_length, embedding_dim}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    embedding_forward_launcher(
        input.data_ptr<int64_t>(),
        weight.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)total_rows,
        (int)embedding_dim,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "embedding_forward CUDA error: ", cudaGetErrorString(err));
    return output;
}

// log_softmax_forward: input CUDA float32 [B,S,D] -> output CUDA float32 [B,S,D]
torch::Tensor log_softmax_forward(torch::Tensor input) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");

    int64_t batch_size = input.size(0);
    int64_t sequence_length = input.size(1);
    int64_t D = input.size(2);
    int64_t total_rows = batch_size * sequence_length;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::empty_like(input, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)total_rows,
        (int)D,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log_softmax_forward CUDA error: ", cudaGetErrorString(err));
    return output;
}

// softmax_forward: input CUDA float32 [B,S,D] -> output CUDA float32 [B,S,D]
torch::Tensor softmax_forward(torch::Tensor input) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");

    int64_t batch_size = input.size(0);
    int64_t sequence_length = input.size(1);
    int64_t D = input.size(2);
    int64_t total_rows = batch_size * sequence_length;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::empty_like(input, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    softmax_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)total_rows,
        (int)D,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "softmax_forward CUDA error: ", cudaGetErrorString(err));
    return output;
}

// kl_div_batchmean_forward: input CUDA float32 [B,S,D], target CUDA float32 [B,S,D] -> output CUDA float32 scalar
torch::Tensor kl_div_batchmean_forward(torch::Tensor input, torch::Tensor target) {
    check_cuda_float(input, "input");
    check_cuda_float(target, "target");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(target.dim() == 3, "target must be 3D");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have same shape");

    int64_t batch_size = input.size(0);
    int64_t total_elements = input.numel();

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor output = torch::empty({}, options);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    kl_div_batchmean_forward_launcher_v2(
        input.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)batch_size,
        (int)total_elements,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kl_div_batchmean_forward CUDA error: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("embedding_forward", &embedding_forward, "Embedding forward",
          py::arg("input"), py::arg("weight"));
    m.def("log_softmax_forward", &log_softmax_forward, "Log softmax forward",
          py::arg("input"));
    m.def("softmax_forward", &softmax_forward, "Softmax forward",
          py::arg("input"));
    m.def("kl_div_batchmean_forward", &kl_div_batchmean_forward, "KL div batchmean forward",
          py::arg("input"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
