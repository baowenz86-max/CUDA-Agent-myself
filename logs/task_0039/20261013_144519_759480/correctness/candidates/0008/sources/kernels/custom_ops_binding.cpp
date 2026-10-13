// Binding file for custom ops: conv2d_bias_bn, avg_pool2d, sigmoid
// No cuDNN dependency; uses custom CUDA kernels from custom_ops.cu.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void conv2d_bias_bn_launcher(
    const float* input,
    int N, int C_in, int H, int W,
    const float* weight,
    const float* conv_bias,
    const float* add_bias,
    const float* bn_weight, const float* bn_bias,
    const float* running_mean, const float* running_var,
    float eps,
    float* output,
    int C_out,
    cudaStream_t stream);

extern "C" void avg_pool2d_launcher(
    const float* input,
    int N, int C, int H, int W,
    int input_n_stride, int input_c_stride, int input_h_stride, int input_w_stride,
    int kernel_size, int stride, int padding,
    float* output, int outH, int outW,
    cudaStream_t stream);

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream);

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while (0)

// Fused Conv2d+bias+BN
// Parameters match model_new.py call:
// conv2d_bias_bn(input, conv_weight, conv_bias, add_bias, bn_weight, bn_bias,
//                running_mean, running_var, eps, kernel_size, stride, padding)
// kernel_size, stride, padding are for avg_pool but unused here; conv uses fixed 3x3 stride 1 padding 1
torch::Tensor conv2d_bias_bn(
    const torch::Tensor& input,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& add_bias,
    const torch::Tensor& bn_weight,
    const torch::Tensor& bn_bias,
    const torch::Tensor& running_mean,
    const torch::Tensor& running_var,
    float eps,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    TORCH_CHECK(add_bias.is_cuda(), "add_bias must be a CUDA tensor");
    TORCH_CHECK(bn_weight.is_cuda(), "bn_weight must be a CUDA tensor");
    TORCH_CHECK(bn_bias.is_cuda(), "bn_bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(conv_weight.dtype() == torch::kFloat32, "conv_weight must be float32");
    TORCH_CHECK(conv_bias.dtype() == torch::kFloat32, "conv_bias must be float32");
    TORCH_CHECK(add_bias.dtype() == torch::kFloat32, "add_bias must be float32");
    TORCH_CHECK(bn_weight.dtype() == torch::kFloat32, "bn_weight must be float32");
    TORCH_CHECK(bn_bias.dtype() == torch::kFloat32, "bn_bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(conv_weight.dim() == 4, "conv_weight must be 4D");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be 1D");
    TORCH_CHECK(add_bias.dim() == 3, "add_bias must be 3D (C,1,1)");
    TORCH_CHECK(bn_weight.dim() == 1, "bn_weight must be 1D");
    TORCH_CHECK(bn_bias.dim() == 1, "bn_bias must be 1D");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");

    TORCH_CHECK(input.size(1) == conv_weight.size(1), "input channels must match conv_weight channels");
    TORCH_CHECK(conv_weight.size(0) == add_bias.size(0), "conv out channels must match add_bias size");
    TORCH_CHECK(conv_weight.size(0) == bn_weight.size(0), "conv out channels must match bn size");

    // Ensure contiguous for kernel (input may have strides, but for now we assume contiguous as typical input)
    auto input_contig = input.contiguous();
    auto weight_contig = conv_weight.contiguous();
    auto conv_bias_contig = conv_bias.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);
    int64_t C_out = weight_contig.size(0);
    // conv kernel size assumed 3 from constructor
    int64_t K = weight_contig.size(2);
    TORCH_CHECK(K == 3, "Only 3x3 conv supported");

    // Output spatial dims (same as input for stride=1,padding=1,kernel=3)
    int64_t H_out = H;
    int64_t W_out = W;

    auto output = torch::empty({N, C_out, H_out, W_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    conv2d_bias_bn_launcher(
        input_contig.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C_in), static_cast<int>(H), static_cast<int>(W),
        weight_contig.data_ptr<float>(),
        conv_bias_contig.data_ptr<float>(),
        add_bias.data_ptr<float>(),
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps,
        output.data_ptr<float>(),
        static_cast<int>(C_out),
        stream);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return output;
}

// AvgPool2d binding (unchanged)
torch::Tensor avg_pool2d(
    const torch::Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    int64_t input_n_stride = input.stride(0);
    int64_t input_c_stride = input.stride(1);
    int64_t input_h_stride = input.stride(2);
    int64_t input_w_stride = input.stride(3);

    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, outH, outW}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avg_pool2d_launcher(
        input.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(input_n_stride), static_cast<int>(input_c_stride), static_cast<int>(input_h_stride), static_cast<int>(input_w_stride),
        static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
        output.data_ptr<float>(), static_cast<int>(outH), static_cast<int>(outW),
        stream);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return output;
}

// Sigmoid binding (unchanged)
torch::Tensor sigmoid(const torch::Tensor& input)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_launcher(input.data_ptr<float>(), output.data_ptr<float>(), static_cast<int>(numel), stream);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_bias_bn", &conv2d_bias_bn,
          "Conv2d + bias + BatchNorm (eval) fused",
          pybind11::arg("input"), pybind11::arg("conv_weight"), pybind11::arg("conv_bias"),
          pybind11::arg("add_bias"), pybind11::arg("bn_weight"), pybind11::arg("bn_bias"),
          pybind11::arg("running_mean"), pybind11::arg("running_var"), pybind11::arg("eps"),
          pybind11::arg("kernel_size"), pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("avg_pool2d", &avg_pool2d, "Average pooling 2d",
          pybind11::arg("input"), pybind11::arg("kernel_size"), pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("sigmoid", &sigmoid, "Element-wise sigmoid", pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);