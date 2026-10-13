#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <vector>
#include "../binding_registry.h"

// -----------------------------------------------------------------------------
// Declarations of launchers from custom_ops.cu
// -----------------------------------------------------------------------------
extern "C" void add_bias_launcher(float* output,
                                  const float* input,
                                  const float* bias,
                                  int channels,
                                  int length,
                                  long long total_elements,
                                  cudaStream_t stream);

extern "C" void roll_launcher(float* output,
                              const float* input,
                              int length,
                              int shifts,
                              long long total_elements,
                              cudaStream_t stream);

extern "C" void view_launcher(float* output,
                              const float* input,
                              long long total_elements,
                              cudaStream_t stream);

extern "C" void linear_launcher(float* output,
                                const float* input,
                                const float* weight,
                                const float* bias,
                                int batch_size,
                                int num_features,
                                int out_features,
                                cudaStream_t stream);

extern "C" void conv_transpose1d_launcher(float* output,
                                          const float* input,
                                          const float* weight,
                                          const float* bias,
                                          int batch_size,
                                          int in_channels,
                                          int out_channels,
                                          int in_length,
                                          int out_length,
                                          int kernel_size,
                                          int stride,
                                          int padding,
                                          cudaStream_t stream);

// -----------------------------------------------------------------------------
// Helper: check CUDA tensor and dtype
// -----------------------------------------------------------------------------
static void check_cuda_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// -----------------------------------------------------------------------------
// 1. ConvTranspose1d forward using custom CUDA kernel
// -----------------------------------------------------------------------------
torch::Tensor conv_transpose1d_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t output_padding) {
    check_cuda_float(input, "input");
    check_cuda_float(weight, "weight");
    check_cuda_float(bias, "bias");

    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "bias out_channels mismatch");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "computed output length must be positive");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    conv_transpose1d_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        batch_size,
        in_channels,
        out_channels,
        input_length,
        output_length,
        kernel_size,
        static_cast<int>(stride),
        static_cast<int>(padding),
        stream);
    return output;
}

// -----------------------------------------------------------------------------
// 2. Add bias forward
// -----------------------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float(input, "input");
    check_cuda_float(bias, "bias");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(bias.dim() == 2 && bias.size(1) == 1, "bias must be [channels, 1]");

    int channels = input.size(1);
    TORCH_CHECK(bias.size(0) == channels, "bias first dim must match channels");

    auto output = torch::empty_like(input);
    long long total_elements = input.numel();

    auto input_contig = input.contiguous();
    auto bias_contig = bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(output.data_ptr<float>(),
                      input_contig.data_ptr<float>(),
                      bias_contig.data_ptr<float>(),
                      channels,
                      input.size(2),
                      total_elements,
                      stream);
    return output;
}

// -----------------------------------------------------------------------------
// 3. Roll forward
// -----------------------------------------------------------------------------
torch::Tensor roll_forward(torch::Tensor input, int64_t shifts, int64_t dims) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(dims == 2, "only dims=2 supported");
    TORCH_CHECK(shifts >= -input.size(2) && shifts <= input.size(2), "shifts out of range");

    auto output = torch::empty_like(input);
    long long total_elements = input.numel();

    auto input_contig = input.contiguous();
    int length = input.size(2);
    int shift_int = static_cast<int>(shifts);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    roll_launcher(output.data_ptr<float>(),
                  input_contig.data_ptr<float>(),
                  length,
                  shift_int,
                  total_elements,
                  stream);
    return output;
}

// -----------------------------------------------------------------------------
// 4. View forward (copy)
// -----------------------------------------------------------------------------
torch::Tensor view_forward(torch::Tensor input, int64_t out_features, int64_t num_features) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.size(0) * input.size(1) * input.size(2) == out_features * num_features,
                "view shape mismatch");

    auto output = torch::empty({input.size(0), num_features}, input.options());
    long long total_elements = input.numel();

    auto input_contig = input.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    view_launcher(output.data_ptr<float>(),
                  input_contig.data_ptr<float>(),
                  total_elements,
                  stream);
    return output;
}

// -----------------------------------------------------------------------------
// 5. Linear forward
// -----------------------------------------------------------------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    check_cuda_float(input, "input");
    check_cuda_float(weight, "weight");
    check_cuda_float(bias, "bias");

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int batch_size = input.size(0);
    int num_features = input.size(1);
    int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == num_features, "weight width must match input features");
    TORCH_CHECK(bias.size(0) == out_features, "bias length must match output features");

    auto output = torch::empty({batch_size, out_features}, input.options());

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_launcher(output.data_ptr<float>(),
                    input_contig.data_ptr<float>(),
                    weight_contig.data_ptr<float>(),
                    bias_contig.data_ptr<float>(),
                    batch_size,
                    num_features,
                    out_features,
                    stream);
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias forward", py::arg("input"), py::arg("bias"));
    m.def("roll_forward", &roll_forward,
          "Roll forward", py::arg("input"), py::arg("shifts"), py::arg("dims"));
    m.def("view_forward", &view_forward,
          "View forward", py::arg("input"), py::arg("out_features"), py::arg("num_features"));
    m.def("linear_forward", &linear_forward,
          "Linear forward", py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
