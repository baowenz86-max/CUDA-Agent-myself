#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    cudaStream_t stream);

extern "C" void sigmoid_forward_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    int64_t input_stride,
    int64_t output_stride,
    cudaStream_t stream);

extern "C" void batch_norm1d_forward_launcher(
    const float* input,
    float* output,
    const float* running_mean,
    const float* running_var,
    const float* weight,
    const float* bias,
    double eps,
    int64_t total_elements,
    int64_t num_features,
    int64_t input_stride,
    int64_t output_stride,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t get_stream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// -----------------------------------------------------------------------------
// linear_forward: output = input @ weight.T + bias
// Parameters:
//   input: [batch, in_features] CUDA float32, contiguous or supports strides (made contiguous)
//   weight: [out_features, in_features] CUDA float32, should be contiguous
//   bias: optional [out_features] CUDA float32
// Returns: [batch, out_features] CUDA float32
// -----------------------------------------------------------------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    // Check CUDA and dtype
    TORCH_CHECK(input.is_cuda(), "linear_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "linear_forward: weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "linear_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "linear_forward: weight must be float32");

    // Ensure contiguous for cuBLAS
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    // Optional bias handling
    const float* bias_ptr = nullptr;
    if (bias.has_value() && bias.value().defined()) {
        auto bias_tensor = bias.value();
        TORCH_CHECK(bias_tensor.is_cuda(), "linear_forward: bias must be a CUDA tensor");
        TORCH_CHECK(bias_tensor.dtype() == torch::kFloat32, "linear_forward: bias must be float32");
        TORCH_CHECK(bias_tensor.dim() == 1 && bias_tensor.size(0) == weight.size(0),
                    "linear_forward: bias shape must be [out_features]");
        auto bias_contig = bias_tensor.contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    // Shape checks
    TORCH_CHECK(input_contig.dim() == 2, "linear_forward: input must be 2D");
    TORCH_CHECK(weight_contig.dim() == 2, "linear_forward: weight must be 2D");
    int64_t batch_size = input_contig.size(0);
    int64_t in_features = input_contig.size(1);
    int64_t out_features = weight_contig.size(0);
    TORCH_CHECK(weight_contig.size(1) == in_features, "linear_forward: weight shape mismatch");

    // Allocate output [batch, out_features]
    auto output = torch::empty({batch_size, out_features}, input_contig.options());

    // Call launcher
    linear_forward_launcher(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch_size,
        in_features,
        out_features,
        get_stream()
    );

    return output;
}

// -----------------------------------------------------------------------------
// sigmoid_forward: element-wise sigmoid
// Parameters:
//   input: any shape CUDA float32, supports strides (we handle stride via launcher)
// Returns: same shape CUDA float32
// -----------------------------------------------------------------------------
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "sigmoid_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "sigmoid_forward: input must be float32");

    // To support arbitrary strides, compute row stride (or element stride if 1D).
    // Since launcher expects stride in elements, we pass input.stride(0) if dim>0 else 1.
    // For contiguous tensors, this will be 1 (assuming last dim contiguous), but if input is non-contiguous
    // the launcher currently assumes linear indexing with stride? We need to check launcher implementation.
    // The launcher uses input[i * input_stride] and output[i * output_stride] with i = total element index.
    // This is incorrect for non-contiguous tensors with non-unit stride in first dimension.
    // To ensure correctness, we simply make input contiguous and pass stride=1.
    auto input_contig = input.contiguous();
    int64_t total_elements = input_contig.numel();

    // Allocate output with same shape
    auto output = torch::empty_like(input_contig);

    // Since we made contiguous, strides are 1 (assuming contiguous memory layout with last-dim stride 1)
    sigmoid_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        1, // input_stride
        1, // output_stride
        get_stream()
    );

    return output;
}

// -----------------------------------------------------------------------------
// batch_norm1d_forward: inference batch norm
// Parameters:
//   input: [batch, num_features] CUDA float32, supports strides (made contiguous)
//   weight: optional [num_features] CUDA float32
//   bias: optional [num_features] CUDA float32
//   running_mean: optional [num_features] CUDA float32
//   running_var: optional [num_features] CUDA float32
//   eps: double
//   momentum: double (unused in inference)
// Returns: [batch, num_features] CUDA float32
// -----------------------------------------------------------------------------
torch::Tensor batch_norm1d_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> running_mean,
    c10::optional<torch::Tensor> running_var,
    double eps,
    double momentum) {

    TORCH_CHECK(input.is_cuda(), "batch_norm1d_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "batch_norm1d_forward: input must be float32");

    // Ensure input is contiguous for simplicity; launcher handles linear indexing with stride=1
    auto input_contig = input.contiguous();
    int64_t batch_size = input_contig.size(0);
    int64_t num_features = input_contig.size(1);
    int64_t total_elements = input_contig.numel();

    // Prepare optional parameter pointers (make contiguous)
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    const float* running_mean_ptr = nullptr;
    const float* running_var_ptr = nullptr;

    // Helper lambda to validate and return contiguous data pointer for optional tensor
    auto get_optional_ptr = [&](const c10::optional<torch::Tensor>& opt, const char* name, int64_t size) -> const float* {
        if (opt.has_value() && opt.value().defined()) {
            auto t = opt.value();
            TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
            TORCH_CHECK(t.dim() == 1 && t.size(0) == size, name, " shape must be [", size, "]");
            return t.contiguous().data_ptr<float>();
        }
        return nullptr;
    };

    weight_ptr = get_optional_ptr(weight, "weight", num_features);
    bias_ptr = get_optional_ptr(bias, "bias", num_features);
    running_mean_ptr = get_optional_ptr(running_mean, "running_mean", num_features);
    running_var_ptr = get_optional_ptr(running_var, "running_var", num_features);

    // Allocate output
    auto output = torch::empty_like(input_contig);

    // Call launcher
    batch_norm1d_forward_launcher(
        input_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        running_mean_ptr,
        running_var_ptr,
        weight_ptr,
        bias_ptr,
        eps,
        total_elements,
        num_features,
        1, // input_stride (since contiguous)
        1, // output_stride
        get_stream()
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward: output = input @ weight.T + bias",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid forward",
          py::arg("input"));
    m.def("batch_norm1d_forward", &batch_norm1d_forward,
          "BatchNorm1d inference forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"),
          py::arg("eps") = 1e-05, py::arg("momentum") = 0.1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
