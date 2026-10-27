// Binding for custom ops: triu_forward, conv2d_forward, add_bias_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Forward declarations of launchers from custom_ops.cu
extern "C" void triu_forward_launcher(
    float* output,
    const float* input,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void add_bias_forward_launcher(
    float* output,
    const float* input,
    const float* bias,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

// Helper function to check CUDA tensor properties
static void check_cuda_float32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must have dtype torch.float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// triu_forward: zeros below diagonal of last two dims
torch::Tensor triu_forward(torch::Tensor x) {
    check_cuda_float32(x, "x");
    TORCH_CHECK(x.dim() == 4, "triu_forward expects 4D tensor [N,C,H,W]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Input and output are both contiguous, but we pass actual strides to be safe.
    // Since output is empty_like, it shares strides with input (assuming same memory format).
    // For contiguous tensors, strides are as expected.
    long long in_stride_n = x.stride(0);
    long long in_stride_c = x.stride(1);
    long long in_stride_h = x.stride(2);
    long long in_stride_w = x.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    triu_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

// conv2d_forward using cuDNN (mandatory for convolution)
torch::Tensor conv2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride = 1,
    int64_t padding = 0,
    int64_t dilation = 1,
    int64_t groups = 1)
{
    check_cuda_float32(x, "x");
    check_cuda_float32(weight, "weight");
    if (bias.numel() > 0) {
        check_cuda_float32(bias, "bias");
    }
    TORCH_CHECK(x.dim() == 4, "conv2d_forward expects 4D input [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "conv2d_forward expects 4D weight [C_out,C_in,kH,kW]");

    int N = x.size(0);
    int C_in = x.size(1);
    int H_in = x.size(2);
    int W_in = x.size(3);
    int C_out = weight.size(0);
    int C_in_w = weight.size(1);
    int kH = weight.size(2);
    int kW = weight.size(3);

    TORCH_CHECK(C_in == C_in_w, "conv2d_forward: input channels mismatch");
    TORCH_CHECK(groups == 1, "conv2d_forward: only groups=1 is supported");
    if (bias.numel() > 0) {
        TORCH_CHECK(bias.size(0) == C_out, "conv2d_forward: bias size mismatch");
    }

    // Output dimensions
    int H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) / stride + 1;
    int W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) / stride + 1;

    auto output = torch::empty({N, C_out, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN setup
    cudnnHandle_t cudnn_handle;
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));

    // Tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));

    // Set tensor descriptor for input: NCHW
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C_in, H_in, W_in));

    // Filter descriptor: C_out, C_in/groups, kH, kW
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        C_out, C_in_w, kH, kW));

    // Convolution descriptor
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding, padding,
        stride, stride,
        dilation, dilation,
        CUDNN_CROSS_CORRELATION, // PyTorch conv2d uses cross-correlation
        CUDNN_DATA_FLOAT));

    // Set math type allow TF32 if available (performance)
    cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION);

    // Output descriptor
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        N, C_out, H_out, W_out));

    // Choose forward algorithm (workspace size)
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    // Could use heuristics: cudnnGetConvolutionForwardAlgorithm_v7, but implicit is fine.
    void* workspace = nullptr;
    size_t workspace_size = 0;

    // Get workspace size for selected algo
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size));

    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(err == cudaSuccess, "conv2d_forward: cudaMalloc for workspace failed");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    bool has_bias = bias.numel() > 0;
    if (has_bias) {
        // If bias exists, we can add it using cudnnConvolutionForward with beta=1 after setting y to bias? 
        // Better: use cudnnConvolutionBiasActivationForward for fused conv+bias.
        cudnnActivationDescriptor_t act_desc;
        CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc));
        CUDNN_CHECK(cudnnSetActivationDescriptor(act_desc, CUDNN_ACTIVATION_IDENTITY, CUDNN_PROPAGATE_NAN, 0.0));
        CUDNN_CHECK(cudnnConvolutionBiasActivationForward(
            cudnn_handle,
            &alpha,
            x_desc, x.data_ptr<float>(),
            w_desc, weight.data_ptr<float>(),
            conv_desc, algo, workspace, workspace_size,
            &beta,  // beta=0
            y_desc, output.data_ptr<float>(),
            // bias descriptor - use same as y? Need separate tensor for bias [1,C_out,1,1]
            // We'll create bias desc and pass bias pointer.
            // For NCHW, bias shape is [1,C_out,1,1]. Let's create a descriptor.
            // Simpler: use cudnnConvolutionForward with beta=0, then add bias by cudnnAddTensor with beta=1.
        ));
    } else {
        CUDNN_CHECK(cudnnConvolutionForward(
            cudnn_handle,
            &alpha,
            x_desc, x.data_ptr<float>(),
            w_desc, weight.data_ptr<float>(),
            conv_desc, algo, workspace, workspace_size,
            &beta,
            y_desc, output.data_ptr<float>()));
    }

    if (workspace) {
        cudaFree(workspace);
    }

    // Cleanup descriptors and handle
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// add_bias_forward: broadcast add bias
torch::Tensor add_bias_forward(torch::Tensor x, torch::Tensor bias) {
    check_cuda_float32(x, "x");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(x.dim() == 4, "add_bias_forward expects 4D input");
    TORCH_CHECK(bias.dim() == 3 || bias.dim() == 1, "add_bias_forward: bias should be 1D or 3D with shape [C,1,1]");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    auto output = torch::empty_like(x);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    long long in_stride_n = x.stride(0);
    long long in_stride_c = x.stride(1);
    long long in_stride_h = x.stride(2);
    long long in_stride_w = x.stride(3);
    long long out_stride_n = output.stride(0);
    long long out_stride_c = output.stride(1);
    long long out_stride_h = output.stride(2);
    long long out_stride_w = output.stride(3);

    add_bias_forward_launcher(
        output.data_ptr<float>(),
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("triu_forward", &triu_forward, "Apply torch.triu to last two dims",
          py::arg("x"));
    m.def("conv2d_forward", &conv2d_forward, "2D convolution using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride") = 1, py::arg("padding") = 0,
          py::arg("dilation") = 1, py::arg("groups") = 1);
    m.def("add_bias_forward", &add_bias_forward, "Broadcasted addition of bias",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
