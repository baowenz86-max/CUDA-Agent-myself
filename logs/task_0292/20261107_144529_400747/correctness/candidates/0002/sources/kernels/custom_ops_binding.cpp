#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// Declare CUDA launchers from custom_ops.cu
extern "C" void batch_norm2d_inference_launcher(
    float* y,
    const float* x,
    const float* weight,
    const float* bias,
    const float* mean,
    const float* var,
    float eps,
    int N, int C, int H, int W,
    cudaStream_t stream);

extern "C" void leaky_relu_launcher(
    float* y,
    const float* x,
    float negative_slope,
    int64_t total_elements,
    cudaStream_t stream);

extern "C" void add_bias_broadcast_launcher(
    float* y,
    const float* x,
    const float* bias,
    int N, int C, int H, int W,
    int64_t x_stride_n,
    int64_t x_stride_c,
    int64_t x_stride_hw,
    cudaStream_t stream);

// cuDNN handle management
static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnCreate(&handle);
    }
    return handle;
}

// Helper to check CUDA tensor properties
static void check_cuda_float32(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must have dtype float32");
}

// -----------------------------------------------------------------------------
// conv2d_no_bias
// -----------------------------------------------------------------------------
torch::Tensor conv2d_no_bias(torch::Tensor x, torch::Tensor weight) {
    check_cuda_float32(x, "x");
    check_cuda_float32(weight, "weight");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (K,C,R,S)");

    // cuDNN requires contiguous tensors for this vintage API
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);
    int K = weight_contig.size(0);
    int R = weight_contig.size(2);
    int S = weight_contig.size(3);
    TORCH_CHECK(weight_contig.size(1) == C, "weight dim1 must match input channels");

    int stride_h = 1;
    int stride_w = 1;
    int pad_h = 1;
    int pad_w = 1;
    int dilation_h = 1;
    int dilation_w = 1;
    int group = 1;

    int H_out = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int W_out = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    auto y = torch::empty({N, K, H_out, W_out}, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Set up cuDNN descriptors
    cudnnHandle_t handle = get_cudnn_handle();
    cudnnSetStream(handle, stream);

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, C, H, W);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               N, K, H_out, W_out);
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               K, C, R, S);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w,
                                    CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(conv_desc, group);

    // Find algorithm (deterministic)
    cudnnConvolutionFwdAlgoPerf_t perf_results[1];
    int returned_algo_count = 0;
    cudnnFindConvolutionForwardAlgorithm(
        handle, x_desc, w_desc, conv_desc, y_desc,
        1, &returned_algo_count, perf_results);

    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;

    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, x_desc, w_desc, conv_desc,
                                            y_desc, algo, &workspace_size);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionForward(handle,
                            &alpha,
                            x_desc, x_contig.data_ptr<float>(),
                            w_desc, weight_contig.data_ptr<float>(),
                            conv_desc, algo,
                            workspace, workspace_size,
                            &beta,
                            y_desc, y.data_ptr<float>());

    if (workspace) {
        cudaFree(workspace);
    }

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return y;
}

// -----------------------------------------------------------------------------
// batch_norm2d_inference
// -----------------------------------------------------------------------------
torch::Tensor batch_norm2d_inference(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    check_cuda_float32(x, "x");
    check_cuda_float32(running_mean, "running_mean");
    check_cuda_float32(running_var, "running_var");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");

    float* weight_ptr = nullptr;
    float* bias_ptr = nullptr;
    if (weight.defined()) {
        check_cuda_float32(weight, "weight");
        TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
        weight_ptr = weight.data_ptr<float>();
    }
    if (bias.defined()) {
        check_cuda_float32(bias, "bias");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
        bias_ptr = bias.data_ptr<float>();
    }

    auto x_contig = x.contiguous();
    auto running_mean_contig = running_mean.contiguous();
    auto running_var_contig = running_var.contiguous();
    torch::Tensor weight_contig;
    torch::Tensor bias_contig;
    if (weight.defined()) weight_contig = weight.contiguous();
    if (bias.defined()) bias_contig = bias.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);
    TORCH_CHECK(running_mean_contig.numel() == C, "running_mean size must match C");
    TORCH_CHECK(running_var_contig.numel() == C, "running_var size must match C");

    auto y = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm2d_inference_launcher(
        y.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        weight_contig.defined() ? weight_contig.data_ptr<float>() : nullptr,
        bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
        running_mean_contig.data_ptr<float>(),
        running_var_contig.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        stream
    );

    return y;
}

// -----------------------------------------------------------------------------
// leaky_relu
// -----------------------------------------------------------------------------
torch::Tensor leaky_relu(torch::Tensor x, double negative_slope) {
    check_cuda_float32(x, "x");
    auto x_contig = x.contiguous();
    int64_t total_elements = x_contig.numel();

    auto y = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    leaky_relu_launcher(
        y.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        static_cast<float>(negative_slope),
        total_elements,
        stream
    );

    return y;
}

// -----------------------------------------------------------------------------
// add_bias_broadcast
// -----------------------------------------------------------------------------
torch::Tensor add_bias_broadcast(torch::Tensor x, torch::Tensor bias) {
    check_cuda_float32(x, "x");
    check_cuda_float32(bias, "bias");
    TORCH_CHECK(x.dim() == 4, "x must be 4D (N,C,H,W)");
    TORCH_CHECK(bias.dim() == 3, "bias must be 3D (C,1,1)");

    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();

    int N = x_contig.size(0);
    int C = x_contig.size(1);
    int H = x_contig.size(2);
    int W = x_contig.size(3);
    TORCH_CHECK(bias_contig.size(0) == C, "bias dim0 must match channels");
    TORCH_CHECK(bias_contig.size(1) == 1 && bias_contig.size(2) == 1, "bias must have shape [C,1,1]");

    auto y = torch::empty_like(x_contig);

    int64_t x_stride_n = (int64_t)C * H * W;
    int64_t x_stride_c = (int64_t)H * W;
    int64_t x_stride_hw = 1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    add_bias_broadcast_launcher(
        y.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        N, C, H, W,
        x_stride_n,
        x_stride_c,
        x_stride_hw,
        stream
    );

    return y;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_no_bias", &conv2d_no_bias, "Conv2d without bias (cuDNN)",
          pybind11::arg("x"), pybind11::arg("weight"));
    m.def("batch_norm2d_inference", &batch_norm2d_inference, "BatchNorm2d inference",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("running_mean"), pybind11::arg("running_var"),
          pybind11::arg("eps") = 1e-05);
    m.def("leaky_relu", &leaky_relu, "LeakyReLU",
          pybind11::arg("x"), pybind11::arg("negative_slope") = 0.1);
    m.def("add_bias_broadcast", &add_bias_broadcast, "Add bias with broadcast",
          pybind11::arg("x"), pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
