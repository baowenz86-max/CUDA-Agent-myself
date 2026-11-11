#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void custom_linear_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int M,
    int K,
    int N,
    bool input_contiguous,
    bool weight_contiguous,
    bool bias_contiguous,
    cudaStream_t stream
);

extern "C" void custom_gelu_launcher(
    float* output,
    const float* input,
    int numel,
    cudaStream_t stream
);

extern "C" void custom_softmax_launcher(
    float* output,
    const float* input,
    int batch,
    int seq_len,
    int features,
    int dim,
    cudaStream_t stream
);

torch::Tensor custom_linear(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    // Validate input
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, seq_len, in_features]");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D [out_features, in_features]");

    int batch = input.size(0);
    int seq_len = input.size(1);
    int in_features = input.size(2);
    int out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");

    torch::Tensor bias_tensor;
    if (bias.has_value() && bias.value().defined()) {
        bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == out_features, "bias shape must be [out_features]");
    }

    // Ensure contiguous layouts
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias_tensor.defined() ? bias_tensor.contiguous() : torch::Tensor();

    // Allocate output
    auto output = torch::empty({batch, seq_len, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr;

    custom_linear_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        batch * seq_len,   // M
        in_features,       // K
        out_features,      // N
        true,              // input_contiguous (always true after .contiguous())
        true,              // weight_contiguous
        bias_ptr != nullptr, // bias_contiguous
        stream
    );

    // Check for CUDA errors (cuBLAS might have reported errors asynchronously)
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in custom_linear: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor custom_gelu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_gelu_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        input_contig.numel(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in custom_gelu: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor custom_softmax(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "softmax currently supports only 3D input");
    TORCH_CHECK(dim == 1 || dim == -3, "only softmax over dim=1 (seq_len) is supported");
    // Normalize negative dim
    int dim_norm = dim < 0 ? dim + input.dim() : dim;
    TORCH_CHECK(dim_norm == 1, "only softmax over dim=1 is supported");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    int batch = input_contig.size(0);
    int seq_len = input_contig.size(1);
    int features = input_contig.size(2);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_softmax_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        batch,
        seq_len,
        features,
        1,  // dim is always 1
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in custom_softmax: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("custom_linear", &custom_linear,
          "Linear transformation (y = x @ weight.T + bias)",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("custom_gelu", &custom_gelu,
          "GELU activation (approximate tanh formula)",
          py::arg("input"));
    m.def("custom_softmax", &custom_softmax,
          "Softmax over given dimension (supports dim=1 only)",
          py::arg("input"), py::arg("dim") = -1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
