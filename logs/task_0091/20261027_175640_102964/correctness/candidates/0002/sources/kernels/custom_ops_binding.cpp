#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void triu_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    long long sN, long long sC, long long sH, long long sW,
    int diagonal,
    cudaStream_t stream);

extern "C" void bias_add_launcher(
    float* x,
    const float* bias,
    int N, int C, int H, int W,
    cudaStream_t stream);

static void check_cuda_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnCreate(&handle);
    }
    return handle;
}

// triu wrapper: applies upper triangular mask to last two dims, supports strides
torch::Tensor triu(torch::Tensor x, int64_t diagonal = 0) {
    check_cuda_float(x, "x");
    TORCH_CHECK(x.dim() == 4, "triu expects 4D input [N,C,H,W]");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(N > 0 && C > 0 && H > 0 && W > 0, "triu input must have positive dimensions");

    // Output is contiguous same shape
    auto out = torch::empty_like(x); // this may not be contiguous? empty_like preserves memory format, but we need contiguous
    out = out.contiguous();

    // Get strides (in elements)
    long long sN = x.stride(0);
    long long sC = x.stride(1);
    long long sH = x.stride(2);
    long long sW = x.stride(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        sN, sC, sH, sW,
        static_cast<int>(diagonal),
        stream
    );

    return out;
}

// conv2d using cuDNN, supports optional bias (nullptr means no bias)
torch::Tensor conv2d(
    torch::Tensor x,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias = c10::nullopt) {
    check_cuda_float(x, "x");
    check_cuda_float(weight, "weight");
    TORCH_CHECK(x.dim() == 4, "conv2d input must be 4D [N,C,H,W]");
    TORCH_CHECK(weight.dim() == 4, "conv2d weight must be 4D [C_out,C_in,kH,kW]");

    int N = x.size(0);
    int C_in = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int C_out = weight.size(0);
    int kH = weight.size(2);
    int kW = weight.size(3);
    TORCH_CHECK(weight.size(1) == C_in, "conv2d weight input channels mismatch");
    TORCH_CHECK(kH > 0 && kW > 0, "kernel size must be positive");

    // Handle optional bias: null means no bias; provided must be [C_out] float32 cuda contiguous
    bool has_bias = bias.has_value() && bias.value().defined();
    if (has_bias) {
        auto b = bias.value();
        check_cuda_float(b, "bias");
        TORCH_CHECK(b.dim() == 1 && b.size(0) == C_out, "bias must be 1D of size C_out");
        TORCH_CHECK(b.is_contiguous(), "bias must be contiguous");
    }

    // cuDNN requires contiguous tensors; we can make x contiguous if needed
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();

    // Dimensions with stride=1, padding=0, dilation=1
    int out_H = H - kH + 1;
    int out_W = W - kW + 1;
    TORCH_CHECK(out_H > 0 && out_W > 0, "conv2d output size must be positive");

    auto y = torch::empty({N, C_out, out_H, out_W}, x.options());

    cudnnHandle_t handle = get_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(handle, stream);

    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors (NCHW)
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, out_H, out_W);
    // Bias descriptor: one per output channel (treated as NCHW with N=1,C_out,1,1)
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1);

    // Filter descriptor: C_out, C_in, kH, kW
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, kH, kW);

    // Convolution descriptor: zero padding, stride 1, dilation 1
    cudnnSetConvolution2dDescriptor(
        conv_desc,
        0, 0,       // pad_height, pad_width
        1, 1,       // stride_height, stride_width
        1, 1,       // dilation_height, dilation_width
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );

    // Find best algorithm (we do forward convolution, then maybe add bias)
    cudnnConvolutionFwdAlgo_t algo;
    constexpr int maxAlgoCount = 8;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[maxAlgoCount];
    cudnnFindConvolutionForwardAlgorithm(
        handle,
        x_desc, w_desc, conv_desc, y_desc,
        maxAlgoCount,
        &returnedAlgoCount,
        perfResults
    );
    TORCH_CHECK(returnedAlgoCount > 0, "cudnnFindConvolutionForwardAlgorithm returned no algorithms");
    algo = perfResults[0].algo;

    // Get workspace size
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size
    );

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(workspace != nullptr, "cudaMalloc failed for cuDNN workspace");
    }

    float alpha = 1.0f, beta = 0.0f;
    // Perform convolution (no activation)
    cudnnStatus_t status = cudnnConvolutionForward(
        handle,
        &alpha,
        x_desc, x_contig.data_ptr<float>(),
        w_desc, weight_contig.data_ptr<float>(),
        conv_desc, algo, workspace, workspace_size,
        &beta,
        y_desc, y.data_ptr<float>()
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionForward failed with status ", status);

    // Add bias if provided
    if (has_bias) {
        beta = 1.0f;
        status = cudnnAddTensor(
            handle,
            &alpha,
            b_desc, bias.value().data_ptr<float>(),
            &beta,
            y_desc, y.data_ptr<float>()
        );
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnAddTensor failed with status ", status);
    }

    // Free workspace
    if (workspace) {
        cudaFree(workspace);
    }

    // Destroy descriptors
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return y;
}

// bias_add: x[b,c,h,w] + bias[c] (bias shape [C,1,1])
torch::Tensor bias_add(torch::Tensor x, torch::Tensor bias) {
    check_cuda_float(x, "x");
    check_cuda_float(bias, "bias");
    TORCH_CHECK(x.dim() == 4, "bias_add expects 4D input [N,C,H,W]");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D [C,1,1]");
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    TORCH_CHECK(bias.size(0) == C && bias.size(1) == 1 && bias.size(2) == 1,
                "bias shape must be [C,1,1]");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    // Allocate output like x (contiguous) and copy x into it (kernel modifies in-place)
    auto y = torch::empty_like(x);
    y = y.contiguous();
    // Copy x to y
    y.copy_(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bias_add_launcher(
        y.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W,
        stream
    );

    return y;
}

void register_custom_ops(pybind11::module& m) {
    m.def("triu", &triu,
          "Apply torch.triu to last two dimensions (CUDA)",
          py::arg("x"), py::arg("diagonal") = 0);
    m.def("conv2d", &conv2d,
          "Conv2d forward using cuDNN (supports optional bias)",
          py::arg("x"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("bias_add", &bias_add,
          "Add channel-wise bias [C,1,1] to input (CUDA)",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
