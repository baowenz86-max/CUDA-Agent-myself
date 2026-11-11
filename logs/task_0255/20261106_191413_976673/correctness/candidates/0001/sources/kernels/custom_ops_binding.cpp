#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// 声明 .cu 中导出的 launcher
extern "C" void avgpool2d_forward_launcher(
    const float* input,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t in_height,
    int64_t in_width,
    int64_t out_height,
    int64_t out_width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width,
    cudaStream_t stream);

extern "C" void add_bias_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int64_t batch,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_height,
    int64_t in_stride_width,
    cudaStream_t stream);

// cuDNN 错误检查辅助函数
#define CUDNN_CHECK(condition) \
    do { \
        cudnnStatus_t status = condition; \
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, \
            "cuDNN error at ", __FILE__, ":", __LINE__, ": ", cudnnGetErrorString(status)); \
    } while (0)

// 获取当前 cuDNN handle（每个线程一个）
static cudnnHandle_t get_cudnn_handle() {
    static thread_local cudnnHandle_t handle = nullptr;
    if (!handle) {
        CUDNN_CHECK(cudnnCreate(&handle));
    }
    return handle;
}

// -----------------------------------------------------------------------------
// avgpool2d_forward 包装
// -----------------------------------------------------------------------------
torch::Tensor avgpool2d_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "avgpool2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "avgpool2d_forward: input must be float32");
    TORCH_CHECK(input.dim() == 4, "avgpool2d_forward: input must be 4D (N,C,H,W)");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t in_height = input.size(2);
    int64_t in_width = input.size(3);

    const int kernel_size = 3;
    const int stride = 3;
    const int padding = 0;
    int64_t out_height = (in_height - kernel_size + 2 * padding) / stride + 1;
    int64_t out_width = (in_width - kernel_size + 2 * padding) / stride + 1;

    // 根据契约，输出为连续 NCHW 张量
    auto output = torch::empty({batch, channels, out_height, out_width}, input.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 如果输入非连续，需要处理 strides；平均池化 kernel 支持任意 strides，
    // 但 launcher 接受 stride 参数。
    // 为了简化并确保正确性，我们直接将输入视为可能非连续，但需要传递 strides。
    // 然而如果 strides 不是正值，可能会有问题。实际 PyTorch 中大多数情况下输入是连续的，
    // 但契约允许非连续，因此我们使用 input.strides() 传递。
    // 注意：input.strides() 返回 IntArrayRef，单位为元素数。
    int64_t in_stride_batch = input.stride(0);
    int64_t in_stride_channel = input.stride(1);
    int64_t in_stride_height = input.stride(2);
    int64_t in_stride_width = input.stride(3);

    // 调用 launcher
    avgpool2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        in_height,
        in_width,
        out_height,
        out_width,
        in_stride_batch,
        in_stride_channel,
        in_stride_height,
        in_stride_width,
        stream
    );

    // 检查 kernel 错误
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "avgpool2d_forward: kernel launch failed");

    return output;
}

// -----------------------------------------------------------------------------
// conv_transpose2d_forward 包装（cuDNN）
// -----------------------------------------------------------------------------
torch::Tensor conv_transpose2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(input.is_cuda(), "conv_transpose2d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose2d_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose2d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose2d_forward: weight must be float32");
    TORCH_CHECK(input.dim() == 4, "conv_transpose2d_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "conv_transpose2d_forward: weight must be 4D");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);

    int64_t C_out = weight.size(1);  // ConvTranspose2d weight shape: [in_channels, out_channels, kH, kW]
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    int64_t H_out = (H_in - 1) * stride - 2 * padding + kH + output_padding;
    int64_t W_out = (W_in - 1) * stride - 2 * padding + kW + output_padding;

    TORCH_CHECK(C_in == weight.size(0), "conv_transpose2d_forward: weight in_channels mismatch");

    // 输出形状
    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    // 获取当前 stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnHandle_t handle = get_cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    // 创建张量描述符
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // 设置张量描述符：NCHW 格式
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)N, (int)C_in, (int)H_in, (int)W_in));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)N, (int)C_out, (int)H_out, (int)W_out));

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        (int)C_in, (int)C_out, (int)kH, (int)kW));

    // 设置卷积描述符：注意对于转置卷积，cuDNN 使用 cudnnConvolutionBackwardData，
    // 需要设置 conv_desc 为对应前向卷积的参数。
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
        (int)padding, (int)padding,
        (int)stride, (int)stride,
        1, 1,  // dilation
        CUDNN_CROSS_CORRELATION,  // PyTorch 使用 cross-correlation
        CUDNN_DATA_FLOAT));

    // 转置卷积相当于在前向卷积的 backward data，
    // 因此使用 cudnnConvolutionBackwardData。
    float alpha = 1.0f, beta = 0.0f;

    // 需要 algo，可以使用 CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 或通过 get 选择
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    // 可能需要工作空间
    size_t workspace_size = 0;
    void* workspace = nullptr;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle, w_desc, x_desc, conv_desc, y_desc, algo, &workspace_size));

    if (workspace_size > 0) {
        // 使用 torch::empty 分配工作空间（仅分配，不计算）
        auto workspace_tensor = torch::empty({static_cast<int64_t>(workspace_size)}, input.options().dtype(torch::kUInt8));
        workspace = workspace_tensor.data_ptr();
        // 注意：workspace_tensor 需要保持存活直到 cuDNN 调用完成，但因为是异步，
        // 我们无法保证其生命周期，因此使用 cudaMalloc 分配更安全。
        // 这里改为 cudaMalloc。
        cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(workspace != nullptr, "conv_transpose2d_forward: failed to allocate workspace");
    }

    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        w_desc, weight.data_ptr<float>(),
        x_desc, input.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        y_desc, output.data_ptr<float>()));

    if (workspace) {
        cudaFree(workspace);
    }

    // 销毁描述符
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "conv_transpose2d_forward: CUDA error");

    return output;
}

// -----------------------------------------------------------------------------
// add_bias_forward 包装
// -----------------------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "add_bias_forward: input must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "add_bias_forward: bias must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "add_bias_forward: input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "add_bias_forward: bias must be float32");
    TORCH_CHECK(input.dim() == 4, "add_bias_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.dim() == 3, "add_bias_forward: bias must be 3D (C,1,1)");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t height = input.size(2);
    int64_t width = input.size(3);

    TORCH_CHECK(channels == bias.size(0), "add_bias_forward: channel dimension mismatch");
    TORCH_CHECK(bias.size(1) == 1 && bias.size(2) == 1, "add_bias_forward: bias must have shape [C,1,1]");

    // 输出与输入具有相同的 shape 和 strides
    auto output = torch::empty_like(input);

    // 获取输入 strides（可能非连续）
    int64_t in_stride_batch = input.stride(0);
    int64_t in_stride_channel = input.stride(1);
    int64_t in_stride_height = input.stride(2);
    int64_t in_stride_width = input.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_bias_forward_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        batch,
        channels,
        height,
        width,
        in_stride_batch,
        in_stride_channel,
        in_stride_height,
        in_stride_width,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_bias_forward: kernel launch failed");

    return output;
}

// -----------------------------------------------------------------------------
// 注册绑定
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("avgpool2d_forward", &avgpool2d_forward, "Average pooling 2d forward (kernel_size=3, stride=3)");
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward, "ConvTranspose2d forward (cuDNN)",
          py::arg("input"), py::arg("weight"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias_forward", &add_bias_forward, "Add bias to input (broadcast along channels)");
}

REGISTER_BINDING(custom_ops, register_custom_ops);
