#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../binding_registry.h"

extern "C" void launch_add_bias(
    float* values, const float* bias, int rows, int columns, cudaStream_t stream);
extern "C" void launch_attention(
    float* query_and_output,
    const float* key,
    const float* value,
    const bool* key_padding_mask,
    int batch_size,
    int sequence_length,
    int embed_dim,
    int num_heads,
    cudaStream_t stream);
extern "C" void launch_copy_and_multiply(
    float* stacked,
    const float* value_input,
    const float* multiplier,
    int count,
    cudaStream_t stream);

namespace {

void check_cuda_float_tensor(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.scalar_type() == torch::kFloat32, name, " must be float32");
}

cublasHandle_t current_cublas_handle(cudaStream_t stream) {
    static thread_local cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        TORCH_CHECK(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate failed");
        TORCH_CHECK(
            cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) == CUBLAS_STATUS_SUCCESS,
            "cublasSetMathMode failed");
    }
    TORCH_CHECK(
        cublasSetStream(handle, stream) == CUBLAS_STATUS_SUCCESS,
        "cublasSetStream failed");
    return handle;
}

void project(
    cublasHandle_t handle,
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int rows,
    int embed_dim,
    cudaStream_t stream) {
    constexpr float alpha = 1.0f;
    constexpr float beta = 0.0f;
    TORCH_CHECK(
        cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            embed_dim,
            rows,
            embed_dim,
            &alpha,
            weight,
            embed_dim,
            input,
            embed_dim,
            &beta,
            output,
            embed_dim) == CUBLAS_STATUS_SUCCESS,
        "cublasSgemm projection failed");
    launch_add_bias(output, bias, rows, embed_dim, stream);
}

}  // namespace

torch::Tensor mha_stack_forward(
    torch::Tensor query_input,
    torch::Tensor key_input,
    torch::Tensor value_input,
    torch::Tensor key_padding_mask,
    torch::Tensor in_projection_weight,
    torch::Tensor in_projection_bias,
    torch::Tensor out_projection_weight,
    torch::Tensor out_projection_bias,
    int64_t num_heads,
    int64_t stack_dim) {
    check_cuda_float_tensor(query_input, "query_input");
    check_cuda_float_tensor(key_input, "key_input");
    check_cuda_float_tensor(value_input, "value_input");
    check_cuda_float_tensor(in_projection_weight, "in_projection_weight");
    check_cuda_float_tensor(in_projection_bias, "in_projection_bias");
    check_cuda_float_tensor(out_projection_weight, "out_projection_weight");
    check_cuda_float_tensor(out_projection_bias, "out_projection_bias");
    TORCH_CHECK(key_padding_mask.is_cuda(), "key_padding_mask must be a CUDA tensor");
    TORCH_CHECK(key_padding_mask.is_contiguous(), "key_padding_mask must be contiguous");
    TORCH_CHECK(key_padding_mask.scalar_type() == torch::kBool, "key_padding_mask must be bool");
    TORCH_CHECK(query_input.dim() == 3, "inputs must have shape [batch, sequence, embedding]");
    TORCH_CHECK(query_input.sizes() == key_input.sizes() && query_input.sizes() == value_input.sizes(), "input shapes must match");
    TORCH_CHECK(stack_dim == 1, "only stack_dim=1 is supported");
    TORCH_CHECK(num_heads > 0, "num_heads must be positive");

    const int batch_size = static_cast<int>(query_input.size(0));
    const int sequence_length = static_cast<int>(query_input.size(1));
    const int embed_dim = static_cast<int>(query_input.size(2));
    const int rows = batch_size * sequence_length;
    TORCH_CHECK(sequence_length == 64, "only sequence_length=64 is supported");
    TORCH_CHECK(embed_dim == 512 && num_heads == 8, "only embed_dim=512 and num_heads=8 are supported");
    TORCH_CHECK(
        key_padding_mask.dim() == 2 && key_padding_mask.size(0) == batch_size &&
            key_padding_mask.size(1) == sequence_length,
        "key_padding_mask has an invalid shape");
    TORCH_CHECK(
        in_projection_weight.dim() == 2 &&
            in_projection_weight.size(0) == 3 * embed_dim &&
            in_projection_weight.size(1) == embed_dim,
        "in_projection_weight has an invalid shape");
    TORCH_CHECK(in_projection_bias.numel() == 3 * embed_dim, "in_projection_bias has an invalid shape");
    TORCH_CHECK(
        out_projection_weight.dim() == 2 &&
            out_projection_weight.size(0) == embed_dim &&
            out_projection_weight.size(1) == embed_dim,
        "out_projection_weight has an invalid shape");
    TORCH_CHECK(out_projection_bias.numel() == embed_dim, "out_projection_bias has an invalid shape");

    auto query_and_attention = torch::empty_like(query_input);
    auto key = torch::empty_like(key_input);
    auto value = torch::empty_like(value_input);
    auto stacked = torch::empty({batch_size, 2, sequence_length, embed_dim}, query_input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cublasHandle_t handle = current_cublas_handle(stream);

    const float* in_weight = in_projection_weight.data_ptr<float>();
    const float* in_bias = in_projection_bias.data_ptr<float>();
    project(
        handle,
        query_input.data_ptr<float>(),
        in_weight,
        in_bias,
        query_and_attention.data_ptr<float>(),
        rows,
        embed_dim,
        stream);
    project(
        handle,
        key_input.data_ptr<float>(),
        in_weight + embed_dim * embed_dim,
        in_bias + embed_dim,
        key.data_ptr<float>(),
        rows,
        embed_dim,
        stream);
    project(
        handle,
        value_input.data_ptr<float>(),
        in_weight + 2 * embed_dim * embed_dim,
        in_bias + 2 * embed_dim,
        value.data_ptr<float>(),
        rows,
        embed_dim,
        stream);

    launch_attention(
        query_and_attention.data_ptr<float>(),
        key.data_ptr<float>(),
        value.data_ptr<float>(),
        key_padding_mask.data_ptr<bool>(),
        batch_size,
        sequence_length,
        embed_dim,
        static_cast<int>(num_heads),
        stream);
    project(
        handle,
        query_and_attention.data_ptr<float>(),
        out_projection_weight.data_ptr<float>(),
        out_projection_bias.data_ptr<float>(),
        stacked.data_ptr<float>() + rows * embed_dim,
        rows,
        embed_dim,
        stream);
    launch_copy_and_multiply(
        stacked.data_ptr<float>(),
        value_input.data_ptr<float>(),
        key_input.data_ptr<float>(),
        rows * embed_dim,
        stream);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");
    return stacked;
}

void register_mha_stack(pybind11::module& module) {
    module.def(
        "mha_stack_forward",
        &mha_stack_forward,
        pybind11::arg("query_input"),
        pybind11::arg("key_input"),
        pybind11::arg("value_input"),
        pybind11::arg("key_padding_mask"),
        pybind11::arg("in_projection_weight"),
        pybind11::arg("in_projection_bias"),
        pybind11::arg("out_projection_weight"),
        pybind11::arg("out_projection_bias"),
        pybind11::arg("num_heads"),
        pybind11::arg("stack_dim"));
}

REGISTER_BINDING(mha_stack, register_mha_stack);
