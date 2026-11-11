#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void launch_relu_forward(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void launch_maxpool2d_forward(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode,
    cudaStream_t stream);

extern "C" void launch_conv2d_forward(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    int groups,
    int out_height,
    int out_width,
    cudaStream_t stream);

// Helper to check CUDA errors
static void checkCuda(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// custom_conv2d using custom CUDA kernel
static torch::Tensor custom_conv2d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int stride,
    int padding,
    int dilation,
    int groups) {

    // Transfer to CUDA if needed
    if (!input.is_cuda()) input = input.cuda();
    if (!weight.is_cuda()) weight = weight.cuda();
    if (!bias.is_cuda()) bias = bias.cuda();

    // Ensure contiguous
    input = input.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    TORCH_CHECK(input.is_cuda(), "custom_conv2d: input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "custom_conv2d: weight must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "custom_conv2d: bias must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_conv2d: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "custom_conv2d: weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "custom_conv2d: bias must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_conv2d: input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "custom_conv2d: weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "custom_conv2d: bias must be 1D");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);
    int out_channels = weight.size(0);
    int kernel_h = weight.size(2);
    int kernel_w = weight.size(3);
    TORCH_CHECK(weight.size(1) == in_channels / groups, "custom_conv2d: weight shape mismatch");
    TORCH_CHECK(bias.numel() == out_channels, "custom_conv2d: bias size mismatch");

    int out_h = (in_h + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    int out_w = (in_w + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

    auto output = torch::empty({batch, out_channels, out_h, out_w}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    launch_conv2d_forward(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        in_channels,
        in_h,
        in_w,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        stride,
        padding,
        padding,
        dilation,
        dilation,
        groups,
        out_h,
        out_w,
        stream
    );
    checkCuda(cudaGetLastError(), "custom_conv2d kernel launch failed");

    return output;
}

static torch::Tensor custom_relu(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "custom_relu: input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_relu: input must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_relu: input must be 4D");
    input = input.contiguous();
    auto output = torch::empty_like(input);
    int64_t total = input.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_relu_forward(input.data_ptr<float>(), output.data_ptr<float>(), total, stream);
    checkCuda(cudaGetLastError(), "custom_relu kernel launch failed");
    return output;
}

static torch::Tensor custom_maxpool2d(
    torch::Tensor input,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    bool ceil_mode) {

    TORCH_CHECK(input.is_cuda(), "custom_maxpool2d: input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_maxpool2d: input must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_maxpool2d: input must be 4D");
    input = input.contiguous();

    int batch = input.size(0);
    int channels = input.size(1);
    int in_h = input.size(2);
    int in_w = input.size(3);

    int out_h, out_w;
    if (ceil_mode) {
        out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        if ((out_h - 1) * stride >= in_h + padding) out_h--;
        if ((out_w - 1) * stride >= in_w + padding) out_w--;
    } else {
        out_h = (in_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        out_w = (in_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    }

    auto output = torch::empty({batch, channels, out_h, out_w}, input.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    launch_maxpool2d_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_h,
        in_w,
        out_h,
        out_w,
        kernel_size,
        stride,
        padding,
        dilation,
        ceil_mode,
        stream
    );
    checkCuda(cudaGetLastError(), "custom_maxpool2d kernel launch failed");
    return output;
}

// Register bindings
static void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv2d", &custom_conv2d, "Custom Conv2d using CUDA kernel",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"), py::arg("groups"));
    m.def("custom_relu", &custom_relu, "Custom ReLU", py::arg("input"));
    m.def("custom_maxpool2d", &custom_maxpool2d, "Custom MaxPool2d",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("ceil_mode"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);