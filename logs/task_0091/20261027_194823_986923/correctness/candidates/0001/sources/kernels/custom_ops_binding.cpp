// Binding file for custom ops: triu_forward, conv2d_forward, bias_add_forward
// Only handles PyTorch tensor wrapping and calls to CUDA launchers or cuDNN.
// No CUDA kernels are defined here.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void triu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long stride_batch, long long stride_channel, long long stride_h, long long stride_w,
    cudaStream_t stream);

extern "C" void bias_add_forward_launcher(
    const float* x,
    const float* bias,
    float* output,
    int N, int C, int H, int W,
    long long stride_batch, long long stride_channel, long long stride_h, long long stride_w,
    cudaStream_t stream);

// Helper: check tensor is CUDA float32
torch::Tensor triu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "triu_forward: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "triu_forward: input must be float32");
    TORCH_CHECK(x.dim() >= 2, "triu_forward: input must have at least 2 dimensions");

    // Output should preserve shape and strides to ensure no layout change
    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    // Prepare dimensions and strides for the last two dims and outer dimensions
    int64_t H = x.size(-2);
    int64_t W = x.size(-1);
    int64_t N = 1, C = 1;
    if (x.dim() == 2) {
        // Treat as shape (H, W) with N=C=1
    } else {
        // For dim > 2, we assume (N, C, H, W) but handle arbitrary leading dims by
        // computing outer product shape. The kernel uses N*C as total outer count,
        // but in the kernel the outer index is i/(H*W), so we need total outer elements.
        // We can pass N*C = outer_elements, and stride_batch = stride of outer index.
        // However the kernel expects N and C separately to compute c and outer.
        // To keep it simple and correct for the known 4D case, pass N and C accordingly.
        if (x.dim() == 4) {
            N = x.size(0);
            C = x.size(1);
        } else {
            // General fallback: flatten all leading dims into batch, C=1
            N = x.numel() / (H * W);
            C = 1;
        }
    }

    // Strides for batch, channel, height, width
    long long stride_batch, stride_channel, stride_h, stride_w;
    if (x.dim() == 4) {
        stride_batch = x.stride(0);
        stride_channel = x.stride(1);
        stride_h = x.stride(2);
        stride_w = x.stride(3);
    } else if (x.dim() == 2) {
        stride_batch = 0;
        stride_channel = 0;
        stride_h = x.stride(0);
        stride_w = x.stride(1);
    } else {
        // General case: assume contiguous-like or last dims stride 1, and compute as if flattened
        stride_batch = H * W;
        stride_channel = H * W;
        stride_h = W;
        stride_w = 1;
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher; for in-place we could pass output and input same, but here output is separate.
    // The launcher does not copy, so we must copy input to output first if pointers differ.
    if (output.data_ptr<float>() != x.data_ptr<float>()) {
        cudaMemcpyAsync(output.data_ptr<float>(), x.data_ptr<float>(),
                        x.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    triu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stride_batch, stride_channel, stride_h, stride_w,
        stream
    );

    return output;
}

torch::Tensor bias_add_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "bias_add_forward: x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias_add_forward: bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "bias_add_forward: x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias_add_forward: bias must be float32");
    TORCH_CHECK(x.dim() == 4, "bias_add_forward: x must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.numel() == x.size(1), "bias_add_forward: bias size must match C");
    // Note: bias may have shape (C,1,1) or (C,), numel==C is enough; we use bias[c] in kernel.

    auto output = torch::empty_strided(x.sizes(), x.strides(), x.options());

    int N = static_cast<int>(x.size(0));
    int C = static_cast<int>(x.size(1));
    int H = static_cast<int>(x.size(2));
    int W = static_cast<int>(x.size(3));

    long long stride_batch = x.stride(0);
    long long stride_channel = x.stride(1);
    long long stride_h = x.stride(2);
    long long stride_w = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_forward_launcher(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        stride_batch, stride_channel, stride_h, stride_w,
        stream
    );

    return output;
}

// cuDNN convolution wrapper
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    TORCH_CHECK(input.is_cuda(), "conv2d_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv2d_forward: weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv2d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv2d_forward: weight must be float32");
    if (bias.has_value()) {
        TORCH_CHECK(bias->is_cuda(), "conv2d_forward: bias must be a CUDA tensor");
        TORCH_CHECK(bias->dtype() == torch::kFloat32, "conv2d_forward: bias must be float32");
        TORCH_CHECK(bias->numel() == weight.size(0), "conv2d_forward: bias size must match out_channels");
    }
    TORCH_CHECK(input.dim() == 4, "conv2d_forward: input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "conv2d_forward: weight must be 4D (out_channels, in_channels/groups, kH, kW)");

    int N = static_cast<int>(input.size(0));
    int C = static_cast<int>(input.size(1));
    int H = static_cast<int>(input.size(2));
    int W = static_cast<int>(input.size(3));
    int K = static_cast<int>(weight.size(0));
    int C_per_group = static_cast<int>(weight.size(1));
    int kH = static_cast<int>(weight.size(2));
    int kW = static_cast<int>(weight.size(3));
    int G = static_cast<int>(groups);

    TORCH_CHECK(C == G * C_per_group, "conv2d_forward: input channels mismatch with groups");

    int outH = (H + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int outW = (W + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    auto output = torch::empty({N, K, outH, outW}, input.options());

    // cuDNN setup
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, c10::cuda::getCurrentCUDAStream().stream());

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors (NHWC vs NCHW; PyTorch uses NCHW)
    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, outH, outW);

    // Bias descriptor: one bias per output channel, shape (1,K,1,1)
    cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K, 1, 1);

    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C_per_group, kH, kW);

    cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding, padding,
        stride, stride,
        dilation, dilation,
        CUDNN_CROSS_CORRELATION, // PyTorch uses cross-correlation
        CUDNN_DATA_FLOAT
    );
    if (G > 1) {
        cudnnSetConvolutionGroupCount(conv_desc, G);
    }

    // Determine algorithm
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, 1, 0, &algo);

    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc, input.data_ptr<float>(),
        filter_desc, weight.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        output_desc, output.data_ptr<float>()
    );

    if (bias.has_value()) {
        // beta = 1 for add bias
        beta = 1.0f;
        cudnnAddTensor(cudnn_handle, &alpha, bias_desc, bias->data_ptr<float>(), &beta, output_desc, output.data_ptr<float>());
        beta = 0.0f; // reset
    }

    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("triu_forward", &triu_forward, "Apply upper triangular mask to last two dimensions",
          py::arg("x"));
    m.def("conv2d_forward", &conv2d_forward, "2D convolution via cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("groups"));
    m.def("bias_add_forward", &bias_add_forward, "Add bias with broadcasting",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
