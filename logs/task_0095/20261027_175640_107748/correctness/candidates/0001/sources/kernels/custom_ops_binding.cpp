#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linspace_launcher(
    float* output,
    float start,
    float end,
    int64_t steps,
    cudaStream_t stream
);

extern "C" void sub_broadcast_launcher(
    float* output,
    const float* x,
    const float* linear_space,
    int64_t batch,
    int64_t channels,
    int64_t length,
    int64_t stride_batch,
    int64_t stride_channel,
    int64_t stride_length,
    cudaStream_t stream
);

extern "C" void conv_transpose1d_launcher(
    const float* input,
    int64_t batch,
    int64_t in_channels,
    int64_t length_in,
    const float* weight,
    int64_t out_channels,
    int64_t kernel_size,
    const float* bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    float* output,
    int64_t length_out,
    cudaStream_t stream
);

// linspace_op: start, end, steps, device (torch.device) -> tensor [steps]
torch::Tensor linspace_op(double start, double end, int64_t steps, torch::Device device) {
    TORCH_CHECK(steps > 0, "steps must be positive");
    TORCH_CHECK(device.is_cuda(), "linspace_op supports only CUDA device");
    
    auto output = torch::empty({steps}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device.index()).stream();
    
    linspace_launcher(
        output.data_ptr<float>(),
        static_cast<float>(start),
        static_cast<float>(end),
        steps,
        stream
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "linspace_launcher failed: ", cudaGetErrorString(err));
    
    return output;
}

// sub_broadcast: x (supports strides), linear_space (contiguous) -> tensor same shape as x
torch::Tensor sub_broadcast(torch::Tensor x, torch::Tensor linear_space) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(linear_space.is_cuda(), "linear_space must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(linear_space.dtype() == torch::kFloat32, "linear_space must be float32");
    TORCH_CHECK(linear_space.is_contiguous(), "linear_space must be contiguous");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [batch, channels, length]");
    TORCH_CHECK(linear_space.dim() == 1, "linear_space must be 1D [length]");
    
    int64_t length = x.size(2);
    TORCH_CHECK(linear_space.size(0) == length, "linear_space length mismatch");
    
    auto output = torch::empty_like(x);
    
    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t stride_batch = x.stride(0);
    int64_t stride_channel = x.stride(1);
    int64_t stride_length = x.stride(2);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    sub_broadcast_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        linear_space.data_ptr<float>(),
        batch,
        channels,
        length,
        stride_batch,
        stride_channel,
        stride_length,
        stream
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "sub_broadcast_launcher failed: ", cudaGetErrorString(err));
    
    return output;
}

// conv_transpose1d: input, weight, bias (optional), stride, padding, output_padding -> tensor
torch::Tensor conv_transpose1d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
    }
    
    TORCH_CHECK(input.dim() == 3, "input must be 3D [batch, in_channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [in_channels, out_channels, kernel_size]");
    
    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t length_in = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);
    
    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    
    if (bias.defined()) {
        TORCH_CHECK(bias.size(0) == out_channels, "bias size mismatch");
    }
    
    int64_t length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(length_out > 0, "output length must be positive");
    
    auto output = torch::empty({batch, out_channels, length_out}, input.options());
    
    const float* bias_ptr = nullptr;
    if (bias.defined()) {
        bias_ptr = bias.data_ptr<float>();
    }
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    conv_transpose1d_launcher(
        input.data_ptr<float>(),
        batch,
        in_channels,
        length_in,
        weight.data_ptr<float>(),
        out_channels,
        kernel_size,
        bias_ptr,
        stride,
        padding,
        output_padding,
        output.data_ptr<float>(),
        length_out,
        stream
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv_transpose1d_launcher failed: ", cudaGetErrorString(err));
    
    return output;
}

void register_custom_ops(pybind11::module& m) {
    pybind11::class_<torch::Device>(m, "Device")
        .def(pybind11::init<>());
    m.def("linspace_op", &linspace_op,
          "linspace_op(start, end, steps, device) -> Tensor",
          pybind11::arg("start"),
          pybind11::arg("end"),
          pybind11::arg("steps"),
          pybind11::arg("device"));
    m.def("sub_broadcast", &sub_broadcast,
          "sub_broadcast(x, linear_space) -> Tensor",
          pybind11::arg("x"),
          pybind11::arg("linear_space"));
    m.def("conv_transpose1d", &conv_transpose1d,
          "conv_transpose1d(input, weight, bias, stride, padding, output_padding) -> Tensor",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
