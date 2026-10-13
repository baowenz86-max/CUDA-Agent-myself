#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void add_bias_launcher(
    float* out,
    const float* in,
    const float* bias,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void batch_norm_eval_launcher(
    float* out,
    const float* in,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void avg_pool2d_launcher(
    float* out,
    const float* in,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_size, int stride, int padding,
    long long in_stride_n, long long in_stride_c, long long in_stride_h, long long in_stride_w,
    long long out_stride_n, long long out_stride_c, long long out_stride_h, long long out_stride_w,
    cudaStream_t stream);

extern "C" void sigmoid_launcher(
    float* out,
    const float* in,
    long long total,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Helper to check cudnn status
static void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// -----------------------------------------------------------------------------
// conv2d_forward using cuDNN
// -----------------------------------------------------------------------------
torch::Tensor conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding) {

    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (C_out, C_in, kH, kW)");
    TORCH_CHECK(input.size(1) == weight.size(1), "input channels must match weight");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == weight.size(0), "bias must be 1D with size equal to output channels");
    }
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    // Make input contiguous if needed (cuDNN requires contiguous)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    // Dimensions
    int N = input_contig.size(0);
    int C_in = input_contig.size(1);
    int H = input_contig.size(2);
    int W = input_contig.size(3);
    int C_out = weight_contig.size(0);
    int kH = weight_contig.size(2);
    int kW = weight_contig.size(3);
    int H_out = (H + 2 * padding - kH) / stride + 1;
    int W_out = (W + 2 * padding - kW) / stride + 1;

    // Allocate output
    auto output = torch::empty({N, C_out, H_out, W_out}, input_contig.options());

    // Get stream
    cudaStream_t stream = getCurrentCUDAStream();

    // cuDNN handles and descriptors
    cudnnHandle_t cudnn_handle;
    checkCudnn(cudnnCreate(&cudnn_handle), "cudnnCreate failed");
    cudnnSetStream(cudnn_handle, stream);

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor input failed");
    checkCudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor output failed");
    checkCudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
    checkCudnn(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor failed");
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");

    // Set tensor descriptors (NCHW format)
    checkCudnn(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W),
               "cudnnSetTensor4dDescriptor input failed");
    checkCudnn(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out),
               "cudnnSetTensor4dDescriptor output failed");
    checkCudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1),
               "cudnnSetTensor4dDescriptor bias failed");

    // Set filter descriptor
    checkCudnn(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, kH, kW),
               "cudnnSetFilter4dDescriptor failed");

    // Set convolution descriptor
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               padding, padding,
                                               stride, stride,
                                               1, 1, // dilation
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT),
               "cudnnSetConvolution2dDescriptor failed");

    // Find best algorithm
    cudnnConvolutionFwdAlgo_t algo;
    checkCudnn(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                   input_desc,
                                                   filter_desc,
                                                   conv_desc,
                                                   output_desc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                   0,
                                                   &algo),
               "cudnnGetConvolutionForwardAlgorithm failed");

    // Workspace size
    size_t workspace_size = 0;
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                       input_desc,
                                                       filter_desc,
                                                       conv_desc,
                                                       output_desc,
                                                       algo,
                                                       &workspace_size),
               "cudnnGetConvolutionForwardWorkspaceSize failed");

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(cuda_err == cudaSuccess, "cudaMalloc workspace failed: ", cudaGetErrorString(cuda_err));
    }

    float alpha = 1.0f, beta = 0.0f;

    // Perform convolution
    checkCudnn(cudnnConvolutionForward(cudnn_handle,
                                       &alpha,
                                       input_desc, input_contig.data_ptr<float>(),
                                       filter_desc, weight_contig.data_ptr<float>(),
                                       conv_desc, algo,
                                       workspace, workspace_size,
                                       &beta,
                                       output_desc, output.data_ptr<float>()),
               "cudnnConvolutionForward failed");

    // Add bias if provided
    if (bias.defined()) {
        // cudnnAddTensor can add bias: out = 1*out + 1*bias
        beta = 1.0f;
        checkCudnn(cudnnAddTensor(cudnn_handle,
                                  &alpha,
                                  bias_desc, bias_contig.data_ptr<float>(),
                                  &beta,
                                  output_desc, output.data_ptr<float>()),
                   "cudnnAddTensor bias failed");
    }

    // Free workspace
    if (workspace != nullptr) {
        cudaFree(workspace);
    }

    // Destroy descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// -----------------------------------------------------------------------------
// add_bias_forward
// -----------------------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(bias.dim() == 3 && bias.size(0) == input.size(1) && bias.size(1) == 1 && bias.size(2) == 1,
                "bias must be shape [C,1,1]");

    // Output contiguous
    auto output = torch::empty_like(input, input.options().memory_format(torch::MemoryFormat::Contiguous));
    auto input_contig = input.contiguous();

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    // Strides for contiguous tensors
    long long in_stride_n = (long long)C * H * W;
    long long in_stride_c = (long long)H * W;
    long long in_stride_h = W;
    long long in_stride_w = 1;
    long long out_stride_n = in_stride_n;
    long long out_stride_c = in_stride_c;
    long long out_stride_h = in_stride_h;
    long long out_stride_w = in_stride_w;

    cudaStream_t stream = getCurrentCUDAStream();
    add_bias_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        bias.data_ptr<float>(),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// batch_norm_eval_forward
// -----------------------------------------------------------------------------
torch::Tensor batch_norm_eval_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
        TORCH_CHECK(weight.dim() == 1 && weight.size(0) == input.size(1), "weight must be shape [C]");
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == input.size(1), "bias must be shape [C]");
    }
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    auto output = torch::empty_like(input, input.options().memory_format(torch::MemoryFormat::Contiguous));
    auto input_contig = input.contiguous();

    const float* weight_ptr = weight.defined() ? weight.data_ptr<float>() : nullptr;
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;

    long long in_stride_n = (long long)C * H * W;
    long long in_stride_c = (long long)H * W;
    long long in_stride_h = W;
    long long in_stride_w = 1;
    long long out_stride_n = in_stride_n;
    long long out_stride_c = in_stride_c;
    long long out_stride_h = in_stride_h;
    long long out_stride_w = in_stride_w;

    cudaStream_t stream = getCurrentCUDAStream();
    batch_norm_eval_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_ptr,
        bias_ptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// avg_pool2d_forward
// -----------------------------------------------------------------------------
torch::Tensor avg_pool2d_forward(
    torch::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(kernel_size > 0, "kernel_size must be positive");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, H_out, W_out}, input.options());
    auto input_contig = input.contiguous();

    long long in_stride_n = (long long)C * H * W;
    long long in_stride_c = (long long)H * W;
    long long in_stride_h = W;
    long long in_stride_w = 1;
    long long out_stride_n = (long long)C * H_out * W_out;
    long long out_stride_c = (long long)H_out * W_out;
    long long out_stride_h = W_out;
    long long out_stride_w = 1;

    cudaStream_t stream = getCurrentCUDAStream();
    avg_pool2d_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        kernel_size, stride, padding,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w,
        stream);

    return output;
}

// -----------------------------------------------------------------------------
// sigmoid_forward
// -----------------------------------------------------------------------------
torch::Tensor sigmoid_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    auto input_contig = input.contiguous();

    long long total = input.numel();

    cudaStream_t stream = getCurrentCUDAStream();
    sigmoid_launcher(output.data_ptr<float>(), input_contig.data_ptr<float>(), total, stream);

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward,
          "Conv2d forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias broadcast",
          pybind11::arg("input"),
          pybind11::arg("bias"));
    m.def("batch_norm_eval_forward", &batch_norm_eval_forward,
          "BatchNorm eval forward",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("running_mean"),
          pybind11::arg("running_var"),
          pybind11::arg("eps"));
    m.def("avg_pool2d_forward", &avg_pool2d_forward,
          "AvgPool2d forward",
          pybind11::arg("input"),
          pybind11::arg("kernel_size"),
          pybind11::arg("stride"),
          pybind11::arg("padding"));
    m.def("sigmoid_forward", &sigmoid_forward,
          "Sigmoid forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
