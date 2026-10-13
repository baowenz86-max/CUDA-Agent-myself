#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" int add_channel_bias_launcher(
    float* output,
    const float* input,
    const float* bias,
    int B,
    int C,
    int L,
    cudaStream_t stream);

extern "C" int round_launcher(
    float* output,
    const float* input,
    int total_elements,
    cudaStream_t stream);

namespace {

// Helper to get cuDNN handle from CUDA stream; uses a per-thread static handle
typedef struct { cudaStream_t stream; cudnnHandle_t handle; } CudnnHandleCache;

cudnnHandle_t get_cudnn_handle(cudaStream_t stream) {
    static thread_local CudnnHandleCache cache = {nullptr, nullptr};
    if (cache.stream != stream || cache.handle == nullptr) {
        if (cache.handle != nullptr) {
            cudnnDestroy(cache.handle);
        }
        CUDNN_CHECK(cudnnCreate(&cache.handle));
        CUDNN_CHECK(cudnnSetStream(cache.handle, stream));
        cache.stream = stream;
    }
    return cache.handle;
}

// Macro for cuDNN error checking; throws on error
typedef struct { const char* msg; } CudnnError;
#define CUDNN_CHECK(expr)                                             \
    do {                                                              \
        cudnnStatus_t status = (expr);                                \
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS,                   \
                    "cuDNN error at ", __FILE__, ":", __LINE__,      \
                    ": " , cudnnGetErrorString(status));                \
    } while (0)

} // anonymous namespace

// ==================== conv_transpose1d_no_bias ====================
torch::Tensor conv_transpose1d_no_bias(
    torch::Tensor input,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding)
{
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, C_in, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(input.size(1) == weight.size(0), "input channels must match weight in_channels");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

    // Make contiguous as cuDNN requires contiguous tensors
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t B = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t L_in = input_contig.size(2);
    int64_t C_out = weight_contig.size(1);
    int64_t K = weight_contig.size(2);

    // Calculate output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "calculated output length must be positive");

    // Allocate output tensor
    auto output = torch::empty({B, C_out, L_out}, input_contig.options());

    // Get current stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnHandle_t handle = get_cudnn_handle(stream);

    // Set tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));

    // Input descriptor: (B, C_in, L_in)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        static_cast<int>(B),
        static_cast<int>(C_in),
        static_cast<int>(L_in),
        1));
    // For 1D convolution, we set height dimension to 1 and width to length
    // Alternative: use 3D descriptor with height=1

    // Output descriptor: (B, C_out, L_out)
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        static_cast<int>(B),
        static_cast<int>(C_out),
        static_cast<int>(L_out),
        1));

    // Filter descriptor: for ConvTranspose, filter is (C_in, C_out, K)
    // cuDNN filter descriptor expects (K, C_in, C_out) when using cudnnSetFilter4dDescriptor
    // For cudnnConvolutionBackwardData, the filter layout is (K, C_out, C_in) when using NCHW data layout
    // We'll use cudnnSetFilter4dDescriptor with dimensions (K, C_in, C_out) and transpose later via mathType
    // Actually, for backward data, the filter is as for the forward convolution: [out_channels, in_channels/groups, kernel_h, kernel_w]
    // But our weight is [in_channels, out_channels, kernel_size]. We need to convert to [K, C_in, C_out] by permuting dims.
    // We'll create a temporary weight tensor permuted to (K, C_in, C_out) contiguous.
    auto weight_permuted = weight_contig.permute({2, 0, 1}).contiguous(); // shape [K, C_in, C_out]

    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        static_cast<int>(C_out),  // out_channels for forward conv
        static_cast<int>(C_in),   // in_channels
        static_cast<int>(1),      // kernel height (unused)
        static_cast<int>(K)));    // kernel width

    // Set convolution descriptor for 2D but with height dims = 1
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        static_cast<int>(padding), // pad_h (but we'll set pad_w; pad_h set to 0)
        static_cast<int>(0),
        static_cast<int>(stride),  // stride_h? Actually for 1D, stride_w
        static_cast<int>(1),
        1, // dilation_h
        1, // dilation_w
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // Correct: for 1D convolution, we use 2D descriptor with height dims of 1 and no padding/stride in height
    // We'll set pad_h = 0, pad_w = padding, stride_h = 1, stride_w = stride
    // But need to be careful with cuDNN tensor layouts: NCHW with H=1, W=length
    // Let's redo conv desc correctly:
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        0, // pad_h
        static_cast<int>(padding), // pad_w
        1, // stride_h
        static_cast<int>(stride), // stride_w
        1, // dilation_h
        1, // dilation_w
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // For transposed convolution, we use cudnnConvolutionBackwardData, which takes
    // dy as input (our x), filter, dx as output (our y).
    // The filter descriptor should be for the forward convolution: output channels -> input channels.
    // But our weight is [C_in, C_out, K]; the forward convolution weight would be [C_out, C_in, K], which is the transpose of our provided weight.
    // However, PyTorch's ConvTranspose1d uses weight [in_channels, out_channels/groups, kernel_size].
    // In cuDNN backward data, the filter is the forward convolution filter, so we need to transpose our weight to [C_out, C_in, K] before setting descriptor? Actually the filter descriptor in cudnnConvolutionBackwardData is the same as for the forward convolution that would have produced dy from dx. The forward convolution's weight would be [C_out, C_in, K]. So we need to pass a filter with that shape. But our weight is [C_in, C_out, K] (for transposed conv). So we need to transpose it to [C_out, C_in, K] before using in filter descriptor.
    // But note: the transposed convolution using backward data is equivalent to the gradient with respect to input of a forward convolution with weight shaped [C_out, C_in, K], stride=stride, padding=padding, dilation=1, and no groups. So this matches PyTorch's transposed conv with weight [C_in, C_out, K] because the forward weight would be the transpose of that (i.e., [C_out, C_in, K]). So we need to permute our weight from (C_in, C_out, K) to (C_out, C_in, K) for the filter descriptor.
    // So we create a permuted weight: shape [C_out, C_in, K] contiguous.
    auto weight_for_cudnn = weight_contig.permute({1, 0, 2}).contiguous(); // [C_out, C_in, K]

    // Now filter descriptor with dimensions [C_out, C_in, 1, K] (for NCHW)
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        static_cast<int>(C_out),   // k = output channels
        static_cast<int>(C_in),    // c = input channels
        static_cast<int>(1),       // r = kernel height (1)
        static_cast<int>(K)));     // s = kernel width

    // Find best algorithm for backward data
    cudnnConvolutionBwdDataAlgo_t algo;
    int requestedAlgoCount = 1, returnedAlgoCount = -1;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(
        handle,
        w_desc,
        y_desc,   // dy descriptor (input to backward data)
        conv_desc,
        x_desc,   // dx descriptor (output of backward data)
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo));

    // Get workspace size
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        w_desc,
        y_desc,
        conv_desc,
        x_desc,
        algo,
        &workspace_size));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(cuda_err == cudaSuccess, "cudaMalloc failed for workspace");
    }

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        w_desc,
        weight_for_cudnn.data_ptr<float>(),
        y_desc,
        input_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        x_desc,
        output.data_ptr<float>()));

    if (workspace != nullptr) {
        cudaFree(workspace);
    }

    CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(w_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    // Check for CUDA errors after kernel launches
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error after conv_transpose");

    return output;
}

// ==================== add_channel_bias ====================
torch::Tensor add_channel_bias(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(input.size(1) == bias.size(0), "bias length must match channel dimension");

    auto input_contig = input.contiguous();
    auto bias_contig = bias.contiguous();
    auto output = torch::empty_like(input_contig);

    int B = static_cast<int>(input_contig.size(0));
    int C = static_cast<int>(input_contig.size(1));
    int L = static_cast<int>(input_contig.size(2));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int status = add_channel_bias_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        B, C, L,
        stream);
    TORCH_CHECK(status == 0, "add_channel_bias kernel launch failed");
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in add_channel_bias");
    return output;
}

// ==================== round ====================
torch::Tensor round(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    int total_elements = static_cast<int>(input_contig.numel());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    int status = round_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        total_elements,
        stream);
    TORCH_CHECK(status == 0, "round kernel launch failed");
    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in round");
    return output;
}

// ==================== Registration ====================
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_no_bias", &conv_transpose1d_no_bias,
          "1D transposed convolution without bias (cuDNN)",
          pybind11::arg("input"), pybind11::arg("weight"),
          pybind11::arg("stride"), pybind11::arg("padding"),
          pybind11::arg("output_padding"));
    m.def("add_channel_bias", &add_channel_bias,
          "Add per-channel bias",
          pybind11::arg("input"), pybind11::arg("bias"));
    m.def("round", &round,
          "Round tensor to nearest integer",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
