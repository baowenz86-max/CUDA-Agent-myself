#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// 声明 .cu 导出的 launcher
extern "C" void reflection_pad2d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int H_in, int W_in,
    int padding,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w,
    cudaStream_t stream);

extern "C" void gather_custom_forward_launcher(
    const float* input,
    const int64_t* indices,
    float* output,
    int N, int C_x, int C_out, int H, int W,
    long long in_stride_n, long long in_stride_c,
    long long in_stride_h, long long in_stride_w,
    long long idx_stride_n, long long idx_stride_c,
    long long idx_stride_h, long long idx_stride_w,
    cudaStream_t stream);

// 辅助函数：检查 CUDA 错误
static void checkCudaError(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

static void checkCudnnError(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

// reflection_pad2d_forward 包装
static torch::Tensor reflection_pad2d_forward_cpp(torch::Tensor x, int64_t padding) {
    // 输入检查
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int N = x.size(0);
    int C = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    // 确保输入是 contiguous，如果不是则先 contiguous，简化 stride 处理
    auto x_contig = x.contiguous();
    
    int H_out = H_in + 2 * (int)padding;
    int W_out = W_in + 2 * (int)padding;
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // 获取 strides（contiguous 后）
    long long in_stride_n = x_contig.stride(0);
    long long in_stride_c = x_contig.stride(1);
    long long in_stride_h = x_contig.stride(2);
    long long in_stride_w = x_contig.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    reflection_pad2d_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H_in, W_in,
        (int)padding,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        stream
    );

    checkCudaError(cudaGetLastError(), "reflection_pad2d_forward_launcher failed");
    return output;
}

// conv2d_cudnn_forward 包装
static torch::Tensor conv2d_cudnn_forward_cpp(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {

    // 参数检查
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(stride >= 1, "stride must be at least 1");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);

    int C_out = weight.size(0);
    int K_h = weight.size(2);
    int K_w = weight.size(3);

    // 确保 weight 和 bias 是 contiguous
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // 计算输出尺寸
    int H_out = (H_in + 2 * (int)padding - K_h) / (int)stride + 1;
    int W_out = (W_in + 2 * (int)padding - K_w) / (int)stride + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    // 获取 stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 创建 cuDNN 句柄
    cudnnHandle_t cudnn_handle;
    checkCudnnError(cudnnCreate(&cudnn_handle), "cudnnCreate failed");
    checkCudnnError(cudnnSetStream(cudnn_handle, stream), "cudnnSetStream failed");

    // 创建张量描述符
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x failed");
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y failed");
    checkCudnnError(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor failed");
    checkCudnnError(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");

    // 设置张量描述符，使用 NCHW 格式
    checkCudnnError(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               N, C_in, H_in, W_in), "cudnnSetTensor4dDescriptor x failed");
    checkCudnnError(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               N, C_out, H_out, W_out), "cudnnSetTensor4dDescriptor y failed");

    // 设置 filter 描述符
    checkCudnnError(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                              C_out, C_in, K_h, K_w), "cudnnSetFilter4dDescriptor failed");

    // 设置卷积描述符
    checkCudnnError(cudnnSetConvolution2dDescriptor(conv_desc,
                                                    (int)padding, (int)padding,
                                                    (int)stride, (int)stride,
                                                    1, 1,
                                                    CUDNN_CROSS_CORRELATION,
                                                    CUDNN_DATA_FLOAT),
                    "cudnnSetConvolution2dDescriptor failed");

    // 查找最快的算法
    int requested_algo_count = 1;
    int returned_algo_count = -1;
    cudnnConvolutionFwdAlgoPerf_t perf_results[1];
    checkCudnnError(cudnnFindConvolutionForwardAlgorithm(
                        cudnn_handle,
                        x_desc, w_desc, conv_desc, y_desc,
                        requested_algo_count, &returned_algo_count, perf_results),
                    "cudnnFindConvolutionForwardAlgorithm failed");
    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    // 获取 workspace 大小
    size_t workspace_size = 0;
    checkCudnnError(cudnnGetConvolutionForwardWorkspaceSize(
                        cudnn_handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size),
                    "cudnnGetConvolutionForwardWorkspaceSize failed");

    // 分配 workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        checkCudaError(cudaMalloc(&workspace, workspace_size), "cudaMalloc workspace failed");
    }

    // 执行卷积
    const float alpha = 1.0f;
    const float beta = 0.0f;  // output 初始为零
    checkCudnnError(cudnnConvolutionForward(
                        cudnn_handle,
                        &alpha,
                        x_desc, x.data_ptr<float>(),
                        w_desc, weight_contig.data_ptr<float>(),
                        conv_desc, algo,
                        workspace, workspace_size,
                        &beta,
                        y_desc, output.data_ptr<float>()),
                    "cudnnConvolutionForward failed");

    // 添加 bias
    if (bias.numel() > 0) {
        // 使用 cudnnAddTensor 或手动 kernel，这里使用 cudnnAddTensor
        cudnnTensorDescriptor_t bias_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
        // bias 是 C_out 维，需要扩展为 NCHW；cudnnAddTensor 支持 broadcast，但需要设置正确的 strides
        checkCudnnError(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                                   1, C_out, 1, 1), "cudnnSetTensor4dDescriptor bias failed");
        const float alpha_bias = 1.0f;
        const float beta_bias = 1.0f;
        checkCudnnError(cudnnAddTensor(cudnn_handle, &alpha_bias, bias_desc, bias_contig.data_ptr<float>(),
                                       &beta_bias, y_desc, output.data_ptr<float>()),
                        "cudnnAddTensor bias failed");
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    // 清理
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    checkCudaError(cudaGetLastError(), "cuda error after conv2d_cudnn_forward");
    return output;
}

// gather_custom_forward 包装
static torch::Tensor gather_custom_forward_cpp(torch::Tensor x, int64_t dim, torch::Tensor indices) {
    // 参数检查
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(indices.dtype() == torch::kInt64, "indices must be int64");
    TORCH_CHECK(x.dim() == 4, "x must be 4D");
    TORCH_CHECK(indices.dim() == 4, "indices must be 4D");
    TORCH_CHECK(dim == 1, "only dim=1 supported");

    int N = x.size(0);
    int C_x = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int C_out = indices.size(1);
    // 检查其他维度匹配
    TORCH_CHECK(indices.size(0) == N, "batch size mismatch");
    TORCH_CHECK(indices.size(2) == H, "height mismatch");
    TORCH_CHECK(indices.size(3) == W, "width mismatch");

    // 确保 x 和 indices contiguous
    auto x_contig = x.contiguous();
    auto indices_contig = indices.contiguous();

    auto output = torch::empty({N, C_out, H, W}, x.options());

    // 获取 strides
    long long in_stride_n = x_contig.stride(0);
    long long in_stride_c = x_contig.stride(1);
    long long in_stride_h = x_contig.stride(2);
    long long in_stride_w = x_contig.stride(3);

    long long idx_stride_n = indices_contig.stride(0);
    long long idx_stride_c = indices_contig.stride(1);
    long long idx_stride_h = indices_contig.stride(2);
    long long idx_stride_w = indices_contig.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    gather_custom_forward_launcher(
        x_contig.data_ptr<float>(),
        indices_contig.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        N, C_x, C_out, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        idx_stride_n, idx_stride_c, idx_stride_h, idx_stride_w,
        stream
    );

    checkCudaError(cudaGetLastError(), "gather_custom_forward_launcher failed");
    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("reflection_pad2d_forward", &reflection_pad2d_forward_cpp,
          "reflection pad 2d forward",
          py::arg("x"), py::arg("padding"));
    m.def("conv2d_cudnn_forward", &conv2d_cudnn_forward_cpp,
          "conv2d cudnn forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 1, py::arg("padding") = 0);
    m.def("gather_custom_forward", &gather_custom_forward_cpp,
          "gather custom forward",
          py::arg("x"), py::arg("dim"), py::arg("indices"));
}

// 自动注册
REGISTER_BINDING(custom_ops, register_custom_ops);
