#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// 声明来自 custom_ops.cu 的 launcher
extern "C" void selu_forward_launcher(
    float* out,
    const float* in,
    int64_t numel,
    cudaStream_t stream
);

namespace {

// 辅助函数：检查 CUDA 错误
void checkCudaError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// cuDNN 错误检查
void checkCudnnError(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

} // anonymous namespace

// selu_forward：对输入张量逐元素应用 SELU
torch::Tensor selu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");

    // 处理非连续输入：SELU kernel 支持任意 stride，但为简单和性能，如果有必要则连续化
    // 注意：如果输入本身就是连续的，则 direct 使用，否则创建一个连续副本
    torch::Tensor x_contig = x.contiguous();

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    selu_forward_launcher(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream
    );

    checkCudaError("selu_forward_launcher failed");

    // 如果输入原本是连续的，output 可以直接返回；否则需要 reshape 回原始形状，但这里形状不变，只是 strides 可能不同
    // 由于 empty_like(x_contig) 结果形状相同，连续性符合要求。
    return output;
}

// conv2d_forward：使用 cuDNN 执行 2D 卷积
torch::Tensor conv2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups
) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(x.dim() == 4, "Input x must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (out_channels, in_channels/groups, kH, kW)");

    // 尺寸信息
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t K = weight.size(0); // out_channels
    int64_t C_per_group = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    TORCH_CHECK(groups >= 1, "groups must be positive");
    TORCH_CHECK(C == C_per_group * groups, "Input channels must equal C_per_group * groups");
    TORCH_CHECK(K % groups == 0, "Output channels must be divisible by groups");

    // 计算输出尺寸（floor 公式）
    int64_t H_out = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    auto output = torch::empty({N, K, H_out, W_out}, x.options());

    // 连续化输入和权重
    torch::Tensor x_contig = x.contiguous();
    torch::Tensor weight_contig = weight.contiguous();

    // 获取当前 CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // 创建 cuDNN handle（每次调用创建/销毁，避免全局状态和线程安全问题，性能影响可接受）
    cudnnHandle_t cudnn_handle;
    checkCudnnError(cudnnCreate(&cudnn_handle), "cudnnCreate failed");
    checkCudnnError(cudnnSetStream(cudnn_handle, stream), "cudnnSetStream failed");

    // 描述符
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnnError(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x failed");
    checkCudnnError(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y failed");
    checkCudnnError(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor failed");
    checkCudnnError(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");

    // 设置张量描述符 (NCHW 格式)
    checkCudnnError(
        cudnnSetTensor4dDescriptor(
            x_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            (int)N,
            (int)C,
            (int)H,
            (int)W
        ),
        "cudnnSetTensor4dDescriptor x failed"
    );
    checkCudnnError(
        cudnnSetTensor4dDescriptor(
            y_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            (int)N,
            (int)K,
            (int)H_out,
            (int)W_out
        ),
        "cudnnSetTensor4dDescriptor y failed"
    );

    // 设置滤波器描述符
    checkCudnnError(
        cudnnSetFilter4dDescriptor(
            w_desc,
            CUDNN_DATA_FLOAT,
            CUDNN_TENSOR_NCHW,
            (int)K,
            (int)C_per_group,
            (int)kH,
            (int)kW
        ),
        "cudnnSetFilter4dDescriptor failed"
    );

    // 设置卷积描述符
    checkCudnnError(
        cudnnSetConvolution2dDescriptor(
            conv_desc,
            (int)padding,
            (int)padding,
            (int)stride,
            (int)stride,
            (int)dilation,
            (int)dilation,
            CUDNN_CROSS_CORRELATION, // PyTorch Conv2d 使用 cross-correlation
            CUDNN_DATA_FLOAT
        ),
        "cudnnSetConvolution2dDescriptor failed"
    );

    // 如果 groups > 1，需要设置 group count (cuDNN 7.0+)
    if (groups > 1) {
        checkCudnnError(
            cudnnSetConvolutionGroupCount(conv_desc, (int)groups),
            "cudnnSetConvolutionGroupCount failed"
        );
    }

    // 选择算法（使用自动调优或默认算法）
    cudnnConvolutionFwdAlgo_t algo;
    // 使用 cudnnGetConvolutionForwardAlgorithm_v7 或 cudnnFindConvolutionForwardAlgorithm
    // 为了简单和可移植性，这里使用默认算法，然后通过 workspace 限制选择最优
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results[requested_algo_count];
    checkCudnnError(
        cudnnFindConvolutionForwardAlgorithm(
            cudnn_handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            requested_algo_count,
            &returned_algo_count,
            perf_results
        ),
        "cudnnFindConvolutionForwardAlgorithm failed"
    );
    TORCH_CHECK(returned_algo_count > 0, "No cuDNN convolution algorithm found");
    algo = perf_results[0].algo;

    // 获取 workspace 大小
    size_t workspace_size = 0;
    checkCudnnError(
        cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_handle,
            x_desc,
            w_desc,
            conv_desc,
            y_desc,
            algo,
            &workspace_size
        ),
        "cudnnGetConvolutionForwardWorkspaceSize failed"
    );

    // 分配 workspace（如果大于 0）
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(cuda_err == cudaSuccess, "cudaMalloc for workspace failed: ", cudaGetErrorString(cuda_err));
    }

    // 确定 bias 指针
    float* bias_ptr = nullptr;
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.value().dim() == 1 && bias.value().size(0) == K, "Bias shape must be [out_channels]");
        bias_contig = bias.value().contiguous();
        bias_ptr = bias_contig.data_ptr<float>();
    }

    // 卷积计算
    float alpha = 1.0f, beta = 0.0f; // 无 bias 时 beta=0，有 bias 时 beta=1 并在之后加 bias，
    // 但 cuDNN 支持融合 bias via cudnnConvolutionBiasActivationForward，这里先使用标准 forward 再手动加 bias
    // 为简化，先用无 bias 阶段，然后单独加 bias kernel？但任务要求不得在 .cpp 中实现 CUDA kernel，加 bias 可以用 cudnnOpTensor 或者直接使用 cudnnAddTensor。
    // 由于我们已经有了 bias 指针，可以使用 cudnnAddTensor 将 bias 加到输出上，此接口不是 CUDA kernel，而是 cuDNN 提供的操作。
    checkCudnnError(
        cudnnConvolutionForward(
            cudnn_handle,
            &alpha,
            x_desc,
            x_contig.data_ptr<float>(),
            w_desc,
            weight_contig.data_ptr<float>(),
            conv_desc,
            algo,
            workspace,
            workspace_size,
            &beta,
            y_desc,
            output.data_ptr<float>()
        ),
        "cudnnConvolutionForward failed"
    );

    // 添加 bias（如果需要）
    if (bias_ptr != nullptr) {
        // 使用 cudnnAddTensor 实现 y += bias
        cudnnTensorDescriptor_t bias_desc;
        checkCudnnError(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
        // bias 是一维的，需要构造为 4D 描述符，将 bias 广播到 NCHW 的输出上
        // cudnnAddTensor 需要 bias 描述符与输出描述符具有相同的维度，但低维可以通过设置 stride 为 0 来实现广播。
        // 简便做法：设置 bias 描述符为 NCHW 形状 [1, K, 1, 1]。
        checkCudnnError(
            cudnnSetTensor4dDescriptor(
                bias_desc,
                CUDNN_TENSOR_NCHW,
                CUDNN_DATA_FLOAT,
                1,
                (int)K,
                1,
                1
            ),
            "cudnnSetTensor4dDescriptor bias failed"
        );
        float alpha_bias = 1.0f;
        checkCudnnError(
            cudnnAddTensor(
                cudnn_handle,
                &alpha_bias,
                bias_desc,
                bias_ptr,
                &alpha_bias, // beta 等于 1，累加到输出
                y_desc,
                output.data_ptr<float>()
            ),
            "cudnnAddTensor bias failed"
        );
        checkCudnnError(cudnnDestroyTensorDescriptor(bias_desc), "cudnnDestroyTensorDescriptor bias failed");
    }

    // 清理
    if (workspace != nullptr) {
        cudaFree(workspace);
    }
    checkCudnnError(cudnnDestroyTensorDescriptor(x_desc), "cudnnDestroyTensorDescriptor x failed");
    checkCudnnError(cudnnDestroyTensorDescriptor(y_desc), "cudnnDestroyTensorDescriptor y failed");
    checkCudnnError(cudnnDestroyFilterDescriptor(w_desc), "cudnnDestroyFilterDescriptor failed");
    checkCudnnError(cudnnDestroyConvolutionDescriptor(conv_desc), "cudnnDestroyConvolutionDescriptor failed");
    checkCudnnError(cudnnDestroy(cudnn_handle), "cudnnDestroy failed");

    checkCudaError("conv2d_forward post-cudnn");
    return output;
}

// 注册函数
void register_custom_ops(pybind11::module& m) {
    m.def("selu_forward", &selu_forward, "SELU activation forward",
          pybind11::arg("x"));
    m.def("conv2d_forward", &conv2d_forward, "2D convolution forward using cuDNN",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("dilation") = 1,
          pybind11::arg("groups") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
