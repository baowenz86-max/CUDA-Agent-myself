#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

namespace py = pybind11;

// -----------------------------------------------------------------------------
// External launcher declarations from custom_ops.cu
// -----------------------------------------------------------------------------
extern "C" void add_bias_forward_launcher(
    const float* input, const float* bias, float* output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride,
    cudaStream_t stream);

extern "C" void round_forward_launcher(
    const float* input, float* output, int total,
    int input_stride, int output_stride, cudaStream_t stream);

extern "C" void triu_forward_launcher(
    const float* input, float* output,
    int batch_size, int out_channels, int length,
    int input_batch_stride, int input_channel_stride,
    int output_batch_stride, int output_channel_stride,
    cudaStream_t stream);

// ----------------------------------------------------------------
// Helper: check CUDA error
// ----------------------------------------------------------------
static void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// ----------------------------------------------------------------
// cuDNN helper for conv_transpose1d
// ----------------------------------------------------------------
static cudnnHandle_t getCudnnHandle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnStatus_t status = cudnnCreate(&handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("cudnnCreate failed");
        }
    }
    return handle;
}

static cudnnTensorDescriptor_t createTensorDesc(
    cudnnTensorFormat_t format,
    cudnnDataType_t dataType,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w)
{
    cudnnTensorDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateTensorDescriptor(&desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnCreateTensorDescriptor failed");
    }
    status = cudnnSetTensor4dDescriptor(
        desc, format, dataType, n, c, h, w);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(desc);
        throw std::runtime_error("cudnnSetTensor4dDescriptor failed");
    }
    // Note: cuDNN doesn't allow setting strides directly via cudnnSetTensor4dDescriptor;
    // for non-contiguous tensors we'll rely on TensorDescriptor with format and dimensions,
    // and ensure tensors are contiguous before calling cuDNN by making contiguous copies if needed.
    // We'll handle that in the wrapper by checking is_contiguous() and calling .contiguous().
    return desc;
}

static cudnnFilterDescriptor_t createFilterDesc(
    cudnnDataType_t dataType,
    cudnnTensorFormat_t format,
    int k, int c, int r, int s)
{
    cudnnFilterDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateFilterDescriptor(&desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnCreateFilterDescriptor failed");
    }
    status = cudnnSetFilter4dDescriptor(
        desc, dataType, format, k, c, r, s);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyFilterDescriptor(desc);
        throw std::runtime_error("cudnnSetFilter4dDescriptor failed");
    }
    return desc;
}

static cudnnConvolutionDescriptor_t createConvDesc(
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    cudnnConvolutionMode_t mode,
    cudnnDataType_t computeType)
{
    cudnnConvolutionDescriptor_t desc;
    cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnCreateConvolutionDescriptor failed");
    }
    status = cudnnSetConvolution2dDescriptor(
        desc,
        pad_h, pad_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        mode,
        computeType);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyConvolutionDescriptor(desc);
        throw std::runtime_error("cudnnSetConvolution2dDescriptor failed");
    }
    return desc;
}

// ----------------------------------------------------------------
// conv_transpose1d_forward
// ----------------------------------------------------------------
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    // Validate tensors
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "bias must be a CUDA tensor if provided");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(!bias.defined() || bias.dtype() == torch::kFloat32, "bias must be float32");

    // Ensure contiguous tensors (cuDNN requires contiguous memory or explicit strides,
    // but we'll use contiguous copies for simplicity)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    // Get dimensions
    int batch_size = input_contig.size(0);
    int in_channels = input_contig.size(1);
    int input_length = input_contig.size(2);

    int weight_in_channels = weight_contig.size(0);
    int out_channels = weight_contig.size(1);
    int kernel_size = weight_contig.size(2);

    TORCH_CHECK(in_channels == weight_in_channels, "input channels and weight in_channels must match");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(output_padding < stride, "output_padding must be less than stride");

    // Compute output length
    int64_t output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "output length must be positive");

    // Create output tensor [batch_size, out_channels, output_length]
    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    // cuDNN expects 4D tensors: we map 1D conv to 2D with height=1
    // Input: [batch, in_channels, 1, input_length] with NCHW format
    // Weight for ConvTranspose1d: shape [in_channels, out_channels, kernel_size]
    // For cuDNN transposed conv, weight shape should be [in_channels, out_channels/k, kh, kw]
    // Here out_channels is the output channels of original conv, but for transposed,
    // cuDNN expects filter shape [in_channels, out_channels, kh, kw].
    // However, PyTorch ConvTranspose1d weight shape is [in_channels, out_channels, kernel_size].
    // We can treat it as [in_channels, out_channels, 1, kernel_size].
    // We'll create a 4D contiguous tensor view for weight without actually copying: reshape to [in_channels, out_channels, 1, kernel_size]
    // But cuDNN filter descriptor expects k (output channels), c (input channels), r, s.
    // For transposed convolution, cuDNN uses the convolution operation with swapped input/output and filter.
    // The filter shape is [out_channels, in_channels/groups, kh, kw] for normal conv, but for transposed,
    // cudnnConvolutionBackwardData (or cudnnConvolutionTranspose) expects filter same as normal conv: [K, C, H, W]
    // where K = out_channels of forward conv = in_channels of transposed? Actually,
    // Let's recall: PyTorch's ConvTranspose1d weight has shape [in_channels, out_channels, kernel_size].
    // But the underlying operation is equivalent to a convolution with weight permuted.
    // cuDNN's cudnnConvolutionBwdData uses filter shape [K, C, H, W] where K is number of output feature maps of the forward convolution,
    // which is the input channels of the transposed convolution? Let's be precise:
    // For a 1D transposed conv with input channels = Ci, output channels = Co, kernel size = K,
    // the equivalent normal convolution (backward data) has filter with shape [Ci, Co, K] if we use the weight as is,
    // because the weight tensor in ConvTranspose1d is actually the same shape as the corresponding Conv1d weight would be,
    // i.e., [in_channels, out_channels, kernel_size]. When computing cudnnConvolutionBackwardData,
    // the filter shape must be [K, C, filter_h, filter_w] where K = number of input feature maps of the forward conv = in_channels,
    // C = output feature maps of forward conv = out_channels? Wait, let's check PyTorch implementation:
    // In PyTorch, ConvTranspose1d internally reshapes weight to [in_channels, out_channels, kernel_size] and passes to cuDNN as
    // cudnnSetFilter4dDescriptor with dims [in_channels, out_channels, kernel_size, 1] for NCHW? Actually,
    // PyTorch's ConvTranspose implementation uses cudnnConvolutionBackwardData with filter representing the kernel of the transposed conv.
    // The filter layout expected by cuDNN for backward data is [K, C, H, W] where K = number of filters (output channels of forward) = in_channels,
    // C = input channels of forward = out_channels. So filter shape must be [in_channels, out_channels, kernel_size, 1].
    // This matches PyTorch's weight shape. So we can directly use weight tensor reshaped to [in_channels, out_channels, kernel_size, 1].
    // We'll create a descriptor with k = weight_contig.size(0), c = weight_contig.size(1), r = weight_contig.size(2), s = 1.

    // Create cuDNN filter descriptor
    cudnnFilterDescriptor_t filter_desc = createFilterDesc(
        CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, weight_contig.size(0), weight_contig.size(1), weight_contig.size(2), 1);

    // Create input tensor descriptor: [batch, in_channels, 1, input_length]
    cudnnTensorDescriptor_t input_desc = createTensorDesc(
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, in_channels, 1, input_length,
        1, 1, 1, 1); // strides will be set if not contiguous but ignored for contiguous case

    // Create output tensor descriptor: [batch, out_channels, 1, output_length]
    cudnnTensorDescriptor_t output_desc = createTensorDesc(
        CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, out_channels, 1, output_length,
        1, 1, 1, 1);

    // Create convolution descriptor: this is for the forward convolution, but we use cudnnConvolutionBackwardData
    cudnnConvolutionDescriptor_t conv_desc = createConvDesc(
        static_cast<int>(padding), 0, // pad_h = padding, pad_w = 0 for 1D (use height=1)
        static_cast<int>(stride), 1,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT);

    // workspace size
    size_t workspace_size = 0;
    cudnnHandle_t handle = getCudnnHandle();
    cudnnStatus_t status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        filter_desc,
        input_desc,
        conv_desc,
        output_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        &workspace_size);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnGetConvolutionBackwardDataWorkspaceSize failed");
    }

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        checkCudaError(cudaMalloc(&workspace, workspace_size), "cudaMalloc workspace");
    }

    // Alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f; // no bias yet; we'll add bias separately if provided, or use beta=1 if bias is provided and we want to add it here.
    // Actually, cuDNN can add bias directly by setting output tensor as input with beta=1 and bias pointer? No, cudnnConvolutionBackwardData doesn't take bias.
    // We'll do conv first, then add bias manually using add_bias if bias is provided.

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(handle, stream);

    // Perform backward data (transposed conv)
    status = cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filter_desc,
        weight_contig.data_ptr<float>(),
        input_desc,
        input_contig.data_ptr<float>(),
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        workspace,
        workspace_size,
        &beta,
        output_desc,
        output.data_ptr<float>());

    // Free workspace
    if (workspace) {
        cudaFree(workspace);
    }
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnConvolutionBackwardData failed");
    }

    // Destroy descriptors
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    // Add bias if provided: use add_bias_forward launcher, but note that bias shape is [out_channels]
    // and we need to add per-channel over length dimension.
    if (bias.defined()) {
        auto bias_contig = bias.contiguous();
        TORCH_CHECK(bias_contig.numel() == out_channels, "bias must have exactly out_channels elements");
        // Launch add_bias_forward with bias viewed as [out_channels, 1]? The launcher expects bias as flat array and
        // uses bias[c] directly, so shape doesn't matter as long as pointer indexed by c.
        add_bias_forward_launcher(
            output.data_ptr<float>(),
            bias_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, out_channels, static_cast<int>(output_length),
            out_channels * static_cast<int>(output_length), static_cast<int>(output_length),
            out_channels * static_cast<int>(output_length), static_cast<int>(output_length),
            stream);
    }

    return output;
}

// ----------------------------------------------------------------
// add_bias_forward
// ----------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");

    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();

    // x shape: [batch_size, out_channels, length]
    int batch_size = x_contig.size(0);
    int out_channels = x_contig.size(1);
    int length = x_contig.size(2);

    TORCH_CHECK(bias_contig.numel() == out_channels, "bias must have out_channels elements");

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_forward_launcher(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, length,
        out_channels * length, length,
        out_channels * length, length,
        stream);

    return output;
}

// ----------------------------------------------------------------
// round_forward
// ----------------------------------------------------------------
torch::Tensor round_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int total = x_contig.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    round_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        1, 1, // contiguous strides
        stream);

    return output;
}

// ----------------------------------------------------------------
// triu_forward
// ----------------------------------------------------------------
torch::Tensor triu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dimensions");

    auto x_contig = x.contiguous();
    int batch_size = x_contig.size(0);
    int out_channels = x_contig.size(1);
    int length = x_contig.size(2);

    auto output = torch::empty_like(x_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    triu_forward_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, out_channels, length,
        out_channels * length, length,
        out_channels * length, length,
        stream);

    return output;
}

// ----------------------------------------------------------------
// Registration
// ----------------------------------------------------------------
void register_custom_ops(py::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward (cuDNN)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias with broadcasting",
          py::arg("x"), py::arg("bias"));
    m.def("round_forward", &round_forward,
          "Round elements (away from zero)",
          py::arg("x"));
    m.def("triu_forward", &triu_forward,
          "Upper triangular for last two dims",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
