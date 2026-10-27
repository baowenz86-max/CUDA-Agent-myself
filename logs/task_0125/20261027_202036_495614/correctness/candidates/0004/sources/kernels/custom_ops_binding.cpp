#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers
extern "C" void batchnorm2d_eval_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float* output,
    int N, int C, int L,
    float eps,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void conv_transpose1d_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int N, int C_in, int C_out, int L_in, int L_out, int K,
    int stride, int padding, int output_padding,
    cudaStream_t stream);

// ConvTranspose1d forward using custom CUDA kernel
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);
    TORCH_CHECK(weight.size(0) == C_in, "weight first dim must match input channels");
    TORCH_CHECK(bias.size(0) == C_out, "bias size must match output channels");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv_transpose1d_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)N, (int)C_in, (int)C_out, (int)L_in, (int)L_out, (int)K,
        (int)stride, (int)padding, (int)output_padding,
        stream
    );

    return output;
}

// BatchNorm2d forward
torch::Tensor batch_norm2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be CUDA");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    TORCH_CHECK(input.size(2) == 1, "third dim must be 1");
    int64_t L = input.size(3);
    TORCH_CHECK(weight.size(0) == C, "weight size mismatch");
    TORCH_CHECK(bias.size(0) == C, "bias size mismatch");
    TORCH_CHECK(running_mean.size(0) == C, "running_mean size mismatch");
    TORCH_CHECK(running_var.size(0) == C, "running_var size mismatch");

    auto output = torch::empty(input.sizes(), input.options());

    long long in_stride_n = input.stride(0);
    long long in_stride_c = input.stride(1);
    long long in_stride_h = input.stride(2);
    long long in_stride_w = input.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batchnorm2d_eval_forward_launcher(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)N, (int)C, (int)L,
        (float)eps,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

// unsqueeze_forward: create a view with an extra dimension of size 1 at dim
torch::Tensor unsqueeze_forward(torch::Tensor input, int64_t dim)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    if (dim < 0) dim += input.dim() + 1;
    TORCH_CHECK(dim >= 0 && dim <= input.dim(), "dim out of range");

    auto sizes = input.sizes().vec();
    auto strides = input.strides().vec();
    sizes.insert(sizes.begin() + dim, 1);
    if (dim == 0) {
        strides.insert(strides.begin(), strides[0] * sizes[1]);
    } else {
        strides.insert(strides.begin() + dim, strides[dim - 1]);
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor view = torch::from_blob(input.data_ptr<float>(), sizes, strides, [](void*) {}, options);
    return view;
}

// squeeze_forward: remove dimension dim if size 1
torch::Tensor squeeze_forward(torch::Tensor input, int64_t dim)
{
    TORCH_CHECK(input.is_cuda(), "input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    if (dim < 0) dim += input.dim();
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim out of range");
    TORCH_CHECK(input.size(dim) == 1, "can only squeeze dimension of size 1");

    auto sizes = input.sizes().vec();
    auto strides = input.strides().vec();
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::Tensor view = torch::from_blob(input.data_ptr<float>(), sizes, strides, [](void*) {}, options);
    return view;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("unsqueeze_forward", &unsqueeze_forward,
          "Unsqueeze forward", py::arg("input"), py::arg("dim"));
    m.def("batch_norm2d_forward", &batch_norm2d_forward,
          "BatchNorm2d eval forward", py::arg("input"), py::arg("weight"),
          py::arg("bias"), py::arg("running_mean"), py::arg("running_var"),
          py::arg("eps"));
    m.def("squeeze_forward", &squeeze_forward,
          "Squeeze forward", py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
