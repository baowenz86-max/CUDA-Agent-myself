#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void leaky_relu_launcher(
    float* out,
    const float* in,
    long long numel,
    int ndim,
    const long long* shape,
    const long long* in_strides,
    const long long* out_strides,
    float negative_slope,
    cudaStream_t stream);

extern "C" void max_pool2d_2x2_stride2_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int outH, int outW,
    const long long* in_strides,
    const long long* out_strides,
    cudaStream_t stream);

// Helper to convert torch::Tensor to vector of int64_t for shape/strides
static std::vector<long long> get_strides(const torch::Tensor& t) {
    std::vector<long long> strides(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        strides[i] = t.stride(i);
    }
    return strides;
}

static std::vector<long long> get_shape(const torch::Tensor& t) {
    std::vector<long long> shape(t.dim());
    for (int i = 0; i < t.dim(); ++i) {
        shape[i] = t.size(i);
    }
    return shape;
}

// Leaky ReLU wrapper
torch::Tensor leaky_relu_forward(torch::Tensor x, double negative_slope = 0.01) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.numel() > 0, "Input tensor must not be empty");

    // Create output with same sizes and strides as input
    auto out = torch::empty_strided(x.sizes(), x.strides(), x.options());

    std::vector<long long> shape = get_shape(x);
    std::vector<long long> in_strides = get_strides(x);
    std::vector<long long> out_strides = get_strides(out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        x.numel(),
        x.dim(),
        shape.data(),
        in_strides.data(),
        out_strides.data(),
        static_cast<float>(negative_slope),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after leaky_relu_launcher: ", cudaGetErrorString(err));

    return out;
}

// MaxPool2d wrapper (fixed kernel_size=2, stride=2, padding=0, dilation=1)
torch::Tensor max_pool2d_forward(torch::Tensor x, int64_t kernel_size = 2, int64_t stride = 2) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_size == 2 && stride == 2, "Only kernel_size=2, stride=2 supported");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int outH = (H - kernel_size) / stride + 1;
    int outW = (W - kernel_size) / stride + 1;

    // Output is contiguous with standard layout [N, C, outH, outW]
    auto out = torch::empty({N, C, outH, outW}, x.options());

    std::vector<long long> in_strides = get_strides(x);
    std::vector<long long> out_strides = get_strides(out);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    max_pool2d_2x2_stride2_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        N, C, H, W,
        outH, outW,
        in_strides.data(),
        out_strides.data(),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after max_pool2d_2x2_stride2_launcher: ", cudaGetErrorString(err));

    return out;
}

// Conv2d wrapper using cuDNN
torch::Tensor conv2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride = 1,
    int64_t dilation = 1)
{
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor if defined");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "Weight must be 4D (K, C, R, S)");
    TORCH_CHECK(weight.size(1) == x.size(1), "Input channels must match weight");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int K = weight.size(0);
    int R = weight.size(2);
    int S = weight.size(3);

    int pad_h = 0;
    int pad_w = 0;
    int stride_h = static_cast<int>(stride);
    int stride_w = static_cast<int>(stride);
    int dilation_h = static_cast<int>(dilation);
    int dilation_w = static_cast<int>(dilation);

    int outH = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int outW = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto out = torch::empty({N, K, outH, outW}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle (per-thread or global? create once maybe better, but for simplicity create per call)
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreate failed");
    cudnnSetStream(cudnn_handle, stream);

    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    status = cudnnCreateTensorDescriptor(&x_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor failed");
    status = cudnnCreateTensorDescriptor(&y_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor failed");
    status = cudnnCreateTensorDescriptor(&b_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor failed");
    status = cudnnCreateFilterDescriptor(&w_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateFilterDescriptor failed");
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateConvolutionDescriptor failed");

    // Set tensor descriptors (NCHW format)
    status = cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor for x failed");
    status = cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, K, outH, outW);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor for y failed");
    status = cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K, 1, 1);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor for bias failed");

    // Set filter descriptor (K, C, R, S)
    status = cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, S);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetFilter4dDescriptor failed");

    // Set convolution descriptor
    status = cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,  // PyTorch uses cross-correlation
        CUDNN_DATA_FLOAT
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetConvolution2dDescriptor failed");

    // Find algorithm (use a simple heuristic; for performance, maybe use cudnnFindConvolutionForwardAlgorithm)
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM; // fallback
    // Try to find best algorithm; but to avoid excessive cost, we can use a fixed algorithm or query.
    // For small conv (3->16, 3x3) implicit GEMM is fine. We'll set to IMPLICIT_GEMM.

    // Allocate workspace if needed (size query)
    size_t workspace_size = 0;
    status = cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle,
        x_desc,
        w_desc,
        conv_desc,
        y_desc,
        algo,
        &workspace_size
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionForwardWorkspaceSize failed");

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(cuda_err == cudaSuccess, "cudaMalloc workspace failed");
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    if (bias.defined()) {
        // Use cudnnConvolutionForward with bias (add bias after conv)
        status = cudnnConvolutionForward(
            cudnn_handle,
            &alpha,
            x_desc, x.data_ptr<float>(),
            w_desc, weight.data_ptr<float>(),
            conv_desc,
            algo,
            workspace, workspace_size,
            &beta,
            y_desc, out.data_ptr<float>()
        );
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionForward failed");

        // Add bias using cudnnAddTensor
        status = cudnnAddTensor(
            cudnn_handle,
            &alpha,
            b_desc, bias.data_ptr<float>(),
            &alpha,  // beta=1 to accumulate on out
            y_desc, out.data_ptr<float>()
        );
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnAddTensor failed");
    } else {
        status = cudnnConvolutionForward(
            cudnn_handle,
            &alpha,
            x_desc, x.data_ptr<float>(),
            w_desc, weight.data_ptr<float>(),
            conv_desc,
            algo,
            workspace, workspace_size,
            &beta,
            y_desc, out.data_ptr<float>()
        );
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionForward failed");
    }

    if (workspace) {
        cudaFree(workspace);
    }

    // Destroy descriptors and handle
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error after conv2d_forward: ", cudaGetErrorString(err));

    return out;
}

void register_custom_ops(pybind11::module& m) {
    m.def("leaky_relu_forward", &leaky_relu_forward,
          "Leaky ReLU forward",
          py::arg("x"),
          py::arg("negative_slope") = 0.01);

    m.def("max_pool2d_forward", &max_pool2d_forward,
          "MaxPool2d forward (kernel=2, stride=2)",
          py::arg("x"),
          py::arg("kernel_size") = 2,
          py::arg("stride") = 2);

    m.def("conv2d_forward", &conv2d_forward,
          "Conv2d forward (cuDNN)",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride") = 1,
          py::arg("dilation") = 1);
}

REGISTER_BINDING(custom_ops, register_custom_ops);