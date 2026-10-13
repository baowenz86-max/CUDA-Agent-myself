#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <ATen/cudnn/cudnn-wrapper.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// 声明 .cu 导出的 launcher
extern "C" void leaky_relu_launcher(
    float* output,
    const float* input,
    int n, int c, int h, int w,
    int in_n_stride, int in_c_stride, int in_h_stride,
    float negative_slope,
    cudaStream_t stream);

extern "C" void add_bias_launcher(
    float* output,
    const float* input,
    const float* bias,
    int n, int c, int h, int w,
    int in_n_stride, int in_c_stride, int in_h_stride,
    cudaStream_t stream);

// 辅助函数：检查张量是否在 CUDA 且为 float32 且非空
static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.numel() > 0, name, " must have at least one element");
}

// conv2d_forward 实现（使用 cuDNN）
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w)
{
    // 参数检查
    check_cuda_float32(input, "input");
    check_cuda_float32(weight, "weight");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_channels, in_channels, kH, kW)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t K = weight.size(0);
    int64_t C_w = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);
    TORCH_CHECK(C == C_w, "input channels must match weight in_channels");
    TORCH_CHECK(bias.numel() == K, "bias size must equal out_channels");

    // 计算输出尺寸
    int64_t out_H = (H + 2 * padding_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t out_W = (W + 2 * padding_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    TORCH_CHECK(out_H > 0 && out_W > 0, "invalid output dimensions");

    // 确保输入、权重、偏置连续，便于传递给 cuDNN
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // 分配输出
    auto output = torch::empty({N, K, out_H, out_W}, input.options());

    // 获取 cuDNN handle
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnCreate(&handle);
    }

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(handle, stream);

    // 创建张量描述符
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // 设置 tensor 描述符
    // NCHW 布局，float32
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, out_H, out_W);
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K, 1, 1);

    // 设置 filter 描述符
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, kH, kW);

    // 设置卷积描述符
    cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding_h, padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,  // PyTorch 默认是 cross-correlation
        CUDNN_DATA_FLOAT);

    // 选择前向算法（使用自动搜索或启发式）
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        1,  // requested algo count
        nullptr,
        &algo);

    // 获取工作空间大小
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        algo,
        &workspace_size);

    // 分配工作空间
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    // 执行卷积
    cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc,
        input_contig.data_ptr<float>(),
        w_desc,
        weight_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        y_desc,
        output.data_ptr<float>());

    // 添加 bias
    cudnnAddTensor(handle, &alpha, b_desc, bias_contig.data_ptr<float>(), &alpha, y_desc, output.data_ptr<float>());

    // 清理
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    // 检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after convolution: ", cudaGetErrorString(err));

    return output;
}

// leaky_relu_forward 实现
torch::Tensor leaky_relu_forward(torch::Tensor input, double negative_slope) {
    check_cuda_float32(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // 获取步长（用于处理非连续张量）
    int64_t in_n_stride = input.stride(0);
    int64_t in_c_stride = input.stride(1);
    int64_t in_h_stride = input.stride(2);
    // w_stride 由 kernel 中 +1 隐式处理

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        N, C, H, W,
        in_n_stride, in_c_stride, in_h_stride,
        static_cast<float>(negative_slope),
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after leaky_relu: ", cudaGetErrorString(err));
    return output;
}

// add_bias_forward 实现
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float32(input, "input");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D");
    TORCH_CHECK(bias.size(0) == input.size(1), "bias channel dimension must match input channels");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "bias spatial dims must be 1");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    // 获取步长
    int64_t in_n_stride = input.stride(0);
    int64_t in_c_stride = input.stride(1);
    int64_t in_h_stride = input.stride(2);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W,
        in_n_stride, in_c_stride, in_h_stride,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after add_bias: ", cudaGetErrorString(err));
    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward,
          "Conv2d forward using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride_h"), py::arg("stride_w"),
          py::arg("padding_h"), py::arg("padding_w"),
          py::arg("dilation_h"), py::arg("dilation_w"));
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          py::arg("input"), py::arg("negative_slope"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias forward",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
