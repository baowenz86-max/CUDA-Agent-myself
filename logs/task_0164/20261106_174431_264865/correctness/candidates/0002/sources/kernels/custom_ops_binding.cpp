#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void maxpool3d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int in_channels,
    int input_depth,
    int input_height,
    int input_width,
    int kernel_size,
    int stride,
    int padding,
    int dilation,
    int ceil_mode,
    cudaStream_t stream);

extern "C" void prod_dim_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream);

// -----------------------------------------------------------------------------
// maxpool3d_forward
// -----------------------------------------------------------------------------
torch::Tensor maxpool3d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    bool ceil_mode)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    // Ensure contiguous for cuDNN (cuDNN requires contiguous in NCDHW)
    input = input.contiguous();

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_depth = input.size(2);
    int input_height = input.size(3);
    int input_width = input.size(4);

    // Compute output dimensions (same formula as cuDNN/ PyTorch floor)
    int output_depth = (input_depth + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_height = (input_height + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int output_width = (input_width + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    TORCH_CHECK(output_depth > 0 && output_height > 0 && output_width > 0, "Invalid pooling output dimensions");

    // Allocate output
    auto output = torch::empty({batch_size, in_channels, output_depth, output_height, output_width}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    maxpool3d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        in_channels,
        input_depth,
        input_height,
        input_width,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        static_cast<int>(ceil_mode),
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "maxpool3d_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// prod_dim_forward
// -----------------------------------------------------------------------------
torch::Tensor prod_dim_forward(
    torch::Tensor input,
    int64_t dim,
    bool keepdim)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 5, "Input must be 5D (N, C, D, H, W)");

    // In this task, dim must be 1 and keepdim true
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");
    TORCH_CHECK(keepdim == true, "Only keepdim=True is supported");

    // Ensure contiguous for simple product kernel
    input = input.contiguous();

    int batch_size = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    // Allocate output: [batch_size, 1, depth, height, width]
    auto output = torch::empty({batch_size, 1, depth, height, width}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    prod_dim_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        channels,
        depth,
        height,
        width,
        stream);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "prod_dim_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("maxpool3d_forward", &maxpool3d_forward,
          "3D max pooling forward (cuDNN)",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"),
          pybind11::arg("ceil_mode"));

    m.def("prod_dim_forward", &prod_dim_forward,
          "Product along dim=1 with keepdim=True",
          pybind11::arg("input"),
          pybind11::arg("dim"),
          pybind11::arg("keepdim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
