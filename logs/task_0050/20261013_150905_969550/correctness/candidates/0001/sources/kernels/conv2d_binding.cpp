// Conv2d binding using cuDNN
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <cudnn.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "../binding_registry.h"

namespace {

// Helper to check cuDNN status
void checkCudnn(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

// Helper to get cuDNN data type from torch dtype
cudnnDataType_t getCudnnDataType(torch::Dtype dtype) {
    if (dtype == torch::kFloat32) return CUDNN_DATA_FLOAT;
    if (dtype == torch::kFloat64) return CUDNN_DATA_DOUBLE;
    if (dtype == torch::kFloat16) return CUDNN_DATA_HALF;
    TORCH_CHECK(false, "Unsupported dtype for cuDNN conv");
}

torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups,
    c10::optional<torch::Tensor> bias_opt) {

    // 检查输入是 CUDA tensor
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    // 检查 dtype 一致（仅支持 float32，按契约）
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");

    // 确保 contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    // 获取维度
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);

    int64_t K = weight_contig.size(0);
    int64_t C_per_group = weight_contig.size(1);
    int64_t R = weight_contig.size(2);
    int64_t S = weight_contig.size(3);

    TORCH_CHECK(C == C_per_group * groups, "input channels must equal weight channels * groups");

    // 输出尺寸计算
    int64_t H_out = (H + 2 * padding_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * padding_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    TORCH_CHECK(H_out > 0 && W_out > 0, "output dimensions must be positive");

    // 可选 bias
    torch::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.size(0) == K, "bias size must match out_channels");
        bias = bias.contiguous();
    }

    // 分配输出
    auto output = torch::empty({N, K, H_out, W_out}, input_contig.options());

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN 句柄
    static cudnnHandle_t handle = nullptr;
    static bool handle_initialized = false;
    if (!handle_initialized) {
        checkCudnn(cudnnCreate(&handle), "cudnnCreate failed");
        handle_initialized = true;
    }
    checkCudnn(cudnnSetStream(handle, stream), "cudnnSetStream failed");

    // 描述符
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnn(cudnnCreateTensorDescriptor(&x_desc), "create x desc");
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc), "create y desc");
    checkCudnn(cudnnCreateTensorDescriptor(&b_desc), "create b desc");
    checkCudnn(cudnnCreateFilterDescriptor(&w_desc), "create w desc");
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "create conv desc");

    // 设置 tensor 描述为 NCHW
    checkCudnn(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         N, C, H, W), "set x desc");
    checkCudnn(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         N, K, H_out, W_out), "set y desc");
    // bias 描述为 1D，按广播到 NCHW 设置
    checkCudnn(cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                         1, K, 1, 1), "set b desc");

    // filter 描述
    checkCudnn(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                         K, C_per_group, R, S), "set w desc");

    // convolution 描述
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               padding_h, padding_w,
                                               stride_h, stride_w,
                                               dilation_h, dilation_w,
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT), "set conv desc");
    if (groups > 1) {
        checkCudnn(cudnnSetConvolutionGroupCount(conv_desc, groups), "set group count");
    }

    // 选择算法
    cudnnConvolutionFwdAlgo_t algo;
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results[requested_algo_count];
    checkCudnn(cudnnFindConvolutionForwardAlgorithm(
        handle,
        x_desc, w_desc, conv_desc, y_desc,
        requested_algo_count, &returned_algo_count, perf_results),
        "cudnnFindConvolutionForwardAlgorithm failed");
    TORCH_CHECK(returned_algo_count > 0, "No convolution algorithm found");
    algo = perf_results[0].algo;

    // 获取工作空间大小
    size_t workspace_size = 0;
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size),
        "cudnnGetConvolutionForwardWorkspaceSize failed");

    // 分配工作空间
    torch::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = torch::empty({static_cast<int64_t>(workspace_size)},
                                 input_contig.options().dtype(torch::kByte));
        workspace_ptr = workspace.data_ptr();
    }

    // 执行卷积
    float alpha = 1.0f;
    float beta = 0.0f;
    if (bias.defined()) {
        beta = 1.0f; // bias 将通过 add tensor 添加
    }

    checkCudnn(cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc, input_contig.data_ptr(),
        w_desc, weight_contig.data_ptr(),
        conv_desc, algo,
        workspace_ptr, workspace_size,
        &beta,
        y_desc, output.data_ptr()),
        "cudnnConvolutionForward failed");

    // 如果有 bias，则添加 bias
    if (bias.defined()) {
        float bias_alpha = 1.0f;
        float bias_beta = 1.0f;
        checkCudnn(cudnnAddTensor(handle, &bias_alpha, b_desc, bias.data_ptr(),
                                  &bias_beta, y_desc, output.data_ptr()),
                   "cudnnAddTensor for bias failed");
    }

    // 销毁描述符
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return output;
}

void register_conv2d(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward,
          "2D convolution forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("stride_h"),
          pybind11::arg("stride_w"),
          pybind11::arg("padding_h"),
          pybind11::arg("padding_w"),
          pybind11::arg("dilation_h"),
          pybind11::arg("dilation_w"),
          pybind11::arg("groups") = 1,
          pybind11::arg("bias") = pybind11::none());
}

} // namespace

REGISTER_BINDING(conv2d, register_conv2d)
