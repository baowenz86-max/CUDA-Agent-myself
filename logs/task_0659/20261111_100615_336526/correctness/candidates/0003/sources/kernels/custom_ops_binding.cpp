#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" {
void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int S, int E,
    int input_stride_0,
    int input_stride_1,
    cudaStream_t stream);

void scaled_dot_product_attention_forward_launcher(
    const float* query,
    const float* key,
    const float* value,
    float* output,
    int N, int S, int E,
    int output_stride_0,
    int output_stride_1,
    int output_stride_2,
    cudaStream_t stream);

void concat_forward_launcher(
    const float* x1,
    const float* x2,
    float* output,
    int N, int S, int E,
    int x1_stride_0, int x1_stride_1,
    int x2_stride_0, int x2_stride_1,
    cudaStream_t stream);
}

// Helper to check CUDA tensor properties
static void check_cuda_float32(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, name, " must be float32");
}

// linear_forward binding
static torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias) {
    check_cuda_float32(input, "input");
    check_cuda_float32(weight, "weight");
    if (bias.has_value()) {
        check_cuda_float32(bias.value(), "bias");
    }

    // Check shapes: input [N, S, E], weight [E, E]
    TORCH_CHECK(input.dim() == 3, "input must be 3D [N, S, E]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [E, E]");
    int N = input.size(0);
    int S = input.size(1);
    int E = input.size(2);
    TORCH_CHECK(weight.size(0) == E && weight.size(1) == E,
                "weight shape must be [", E, ", ", E, "]");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().dim() == 1 && bias.value().size(0) == E,
                    "bias must be 1D of size ", E);
    }

    // Ensure weight is contiguous for cuBLAS (will likely be contiguous already)
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.has_value() ? bias.value().contiguous() : torch::Tensor();

    // Allocate output as contiguous [N, S, E]
    auto output = torch::empty({N, S, E}, input.options());

    // Get stream
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    // Gather input stride info for logical [N*S, E] matrix
    int input_stride_0 = input.stride(0); // stride between batches
    int input_stride_1 = input.stride(2); // stride between elements within a row

    // Call launcher
    linear_forward_launcher(
        input.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias.has_value() ? bias_contig.data_ptr<float>() : nullptr,
        output.data_ptr<float>(),
        N, S, E,
        input_stride_0, input_stride_1,
        stream);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "linear_forward kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

// scaled_dot_product_attention_forward binding
static torch::Tensor scaled_dot_product_attention_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value) {
    check_cuda_float32(query, "query");
    check_cuda_float32(key, "key");
    check_cuda_float32(value, "value");

    // Shapes: [N, S, E]
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
                "query/key/value must be 3D [N, S, E]");
    int N = query.size(0);
    int S = query.size(1);
    int E = query.size(2);
    TORCH_CHECK(key.size(0) == N && key.size(1) == S && key.size(2) == E,
                "key shape mismatch");
    TORCH_CHECK(value.size(0) == N && value.size(1) == S && value.size(2) == E,
                "value shape mismatch");

    // Ensure inputs are contiguous (launcher assumes contiguous query/key/value)
    auto query_contig = query.contiguous();
    auto key_contig = key.contiguous();
    auto value_contig = value.contiguous();

    // Allocate output with special strides [E, S*E, 1]
    int out_stride_0 = E;
    int out_stride_1 = S * E;
    int out_stride_2 = 1;

    // Create output tensor with these strides.
    // Use torch::empty and as_strided; ensure enough storage.
    auto output = torch::empty({N, S, E}, query.options()).as_strided(
        {N, S, E}, {out_stride_0, out_stride_1, out_stride_2});

    // Get stream
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    scaled_dot_product_attention_forward_launcher(
        query_contig.data_ptr<float>(),
        key_contig.data_ptr<float>(),
        value_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, S, E,
        out_stride_0, out_stride_1, out_stride_2,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "scaled_dot_product_attention_forward kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

// concat_forward binding
static torch::Tensor concat_forward(
    torch::Tensor x1,
    torch::Tensor x2) {
    check_cuda_float32(x1, "x1");
    check_cuda_float32(x2, "x2");

    // Shapes: [N, S, E]
    TORCH_CHECK(x1.dim() == 3 && x2.dim() == 3,
                "x1/x2 must be 3D [N, S, E]");
    int N = x1.size(0);
    int S = x1.size(1);
    int E = x1.size(2);
    TORCH_CHECK(x2.size(0) == N && x2.size(1) == S && x2.size(2) == E,
                "x2 shape mismatch");

    // Gather stride info for 3D indexing
    int x1_stride_0 = x1.stride(0);
    int x1_stride_1 = x1.stride(1);
    int x2_stride_0 = x2.stride(0);
    int x2_stride_1 = x2.stride(1);

    // Allocate output contiguous [N, S, 2E]
    auto output = torch::empty({N, S, 2 * E}, x1.options());

    // Get stream
    auto stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    concat_forward_launcher(
        x1.data_ptr<float>(),
        x2.data_ptr<float>(),
        output.data_ptr<float>(),
        N, S, E,
        x1_stride_0, x1_stride_1,
        x2_stride_0, x2_stride_1,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "concat_forward kernel launch failed: ",
                cudaGetErrorString(err));

    return output;
}

// Register bindings
static void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward (input @ weight.T + bias)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("scaled_dot_product_attention_forward", &scaled_dot_product_attention_forward,
          "Scaled dot-product attention forward",
          py::arg("query"), py::arg("key"), py::arg("value"));
    m.def("concat_forward", &concat_forward,
          "Concatenate along last dimension",
          py::arg("x1"), py::arg("x2"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
