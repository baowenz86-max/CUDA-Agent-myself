#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Launcher declaration from custom_ops.cu
extern "C" void sum_dim_launcher(float* output, const float* input,
                                 int dim, int batch, int channels, int length,
                                 long long in_stride_batch, long long in_stride_channel, long long in_stride_length,
                                 long long out_stride_batch, long long out_stride_length,
                                 cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while (0)

// Helper to check cuDNN errors
#define CHECK_CUDNN(call) \
    do { \
        cudnnStatus_t status = call; \
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(status)); \
    } while (0)

// Get cuDNN data type for float tensors
cudnnDataType_t getCudnnDataType(torch::Tensor tensor) {
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Only float32 tensors are supported");
    return CUDNN_DATA_FLOAT;
}

// Ensure tensor is contiguous and on CUDA
torch::Tensor ensure_contiguous_cuda(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(tensor.dtype() == torch::kFloat32, "Tensor must be float32");
    return tensor.contiguous();
}

// ===================== conv_transpose1d_forward =====================
torch::Tensor conv_transpose1d_forward(torch::Tensor input, torch::Tensor weight,
                                       c10::optional<torch::Tensor> bias, int64_t stride,
                                       int64_t padding, int64_t output_padding) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input dtype must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight dtype must be float32");

    // Ensure contiguity
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias dtype must be float32");
        bias_contig = bias.value().contiguous();
    }

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(1);
    int64_t kernel_size = weight_contig.size(2);

    // Check weight shape compatibility
    TORCH_CHECK(weight_contig.size(0) == in_channels, "weight first dimension must match input channels");

    // Compute output length
    int64_t output_length = (length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "output length must be positive");

    // Allocate output
    auto output = torch::empty({batch, out_channels, output_length}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle (static to avoid repeated creation)
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUDNN(cudnnCreate(&handle));
    }

    // Set stream
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Describe tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors (NCHW format, but we pretend HW is length and 1)
    // For 1D convolution, cuDNN uses 2D descriptors with height=1, width=length.
    int dims[4] = {static_cast<int>(batch), static_cast<int>(in_channels), 1, static_cast<int>(length)};
    int strides[4] = {static_cast<int>(in_channels * length), static_cast<int>(length), static_cast<int>(length), 1};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           dims[0], dims[1], dims[2], dims[3]));

    int out_dims[4] = {static_cast<int>(batch), static_cast<int>(out_channels), 1, static_cast<int>(output_length)};
    int out_strides[4] = {static_cast<int>(out_channels * output_length), static_cast<int>(output_length), static_cast<int>(output_length), 1};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           out_dims[0], out_dims[1], out_dims[2], out_dims[3]));

    // Set bias descriptor (one bias per output channel)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, static_cast<int>(out_channels), 1, 1));

    // Set filter descriptor: [in_channels, out_channels, 1, kernel_size]
    // cuDNN expects filter as [K, C, R, S] for NCHW, i.e., out_channels, in_channels, height, width
    // But PyTorch weight is [in_channels, out_channels, kernel_size].
    // For transposed convolution, cuDNN uses filter layout [in_channels, out_channels, kernel_size] as well?
    // Actually, for cudnnConvolutionBackwardData (which implements transposed conv),
    // the filter is the same as the forward conv filter: [out_channels, in_channels, kernel_size].
    // PyTorch ConvTranspose1d weight is [in_channels, out_channels, kernel_size], so we need to
    // treat it as a transposed filter. We can use cudnnSetFilter4dDescriptor with appropriate dimensions
    // and pass the weight data pointer. However, because of the weight layout, we may need to use
    // cudnnSetFilter4dDescriptor with format CUDNN_TENSOR_NCHW and dims [out_channels, in_channels, 1, kernel_size]
    // but the memory layout is [in_channels, out_channels, kernel_size], which is not contiguous in that order.
    // Instead, we can create a contiguous copy of weight in [out_channels, in_channels, 1, kernel_size] layout.
    // Alternatively, we can use cudnnSetFilter4dDescriptor with CUDNN_TENSOR_NCHW and dims
    // [in_channels, out_channels, 1, kernel_size]? cuDNN expects filter tensor dims as [K, C, R, S] where
    // K = number of output feature maps, C = number of input feature maps, R = filter height, S = filter width.
    // For transposed convolution via cudnnConvolutionBackwardData, the filter is the same as forward conv:
    // [K, C, R, S] = [out_channels, in_channels, 1, kernel_size].
    // But PyTorch ConvTranspose1d weight shape is [in_channels, out_channels, kernel_size].
    // To match, we need to transpose the weight to [out_channels, in_channels, kernel_size] first.
    // We can either use a temporary transposed tensor or use cuDNN's ability to handle strided filter descriptors.
    // For simplicity, we create a transposed contiguous copy.
    auto weight_transposed = weight_contig.permute({1, 0, 2}).contiguous();
    // Now weight_transposed shape: [out_channels, in_channels, kernel_size]
    // We'll treat it as a 4D filter with height=1, width=kernel_size.
    int filter_dims[4] = {static_cast<int>(out_channels), static_cast<int>(in_channels), 1, static_cast<int>(kernel_size)};
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]));

    // Set convolution descriptor for transposed convolution
    // For cudnnConvolutionBackwardData, we need to set the convolution descriptor with the same parameters
    // as the forward conv that would have produced this gradient. That means:
    // conv_desc describes a convolution with stride=stride, pad=padding, dilation=1.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding, padding,  // pad height, width (top/bottom, left/right)
                                                stride, stride,    // stride height, width
                                                1, 1,              // dilation height, width
                                                CUDNN_CROSS_CORRELATION,  // mode
                                                CUDNN_DATA_FLOAT));

    // Choose algorithm (we'll use a simple default; for better performance could use cudnnFindConvolutionBackwardDataAlgorithm)
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    // Get workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                              filter_desc,
                                                              output_desc,  // note: in backward data, output_desc is the gradient wrt input, i.e., the transposed conv output? Actually, need to be careful.
                                                              conv_desc,
                                                              input_desc,  // this is the input to the transposed conv, i.e., the "dy"
                                                              algo,
                                                              &workspace_size));

    // For cudnnConvolutionBackwardData, the parameters are:
    // cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, algo, workspace, workspaceSize, beta, dxDesc, dx)
    // where dx is the output of transposed conv (the "data gradient"), dy is the input to transposed conv (the "output gradient"),
    // and w is the filter. So we set:
    // dy = input_contig (our transposed conv input)
    // w = weight_transposed (filter)
    // dx = output (our transposed conv output)
    // The descriptors: dyDesc = input_desc (batch, in_channels, length), dxDesc = output_desc (batch, out_channels, output_length),
    // wDesc = filter_desc (out_channels, in_channels, kernel_size).
    // This matches PyTorch ConvTranspose1d.

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(handle,
                                              &alpha,
                                              filter_desc, weight_transposed.data_ptr<float>(),
                                              input_desc, input_contig.data_ptr<float>(),
                                              conv_desc,
                                              algo,
                                              workspace, workspace_size,
                                              &beta,
                                              output_desc, output.data_ptr<float>()));

    // Add bias if present
    if (bias_contig.defined()) {
        float bias_alpha = 1.0f, bias_beta = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(handle,
                                    &bias_alpha,
                                    bias_desc, bias_contig.data_ptr<float>(),
                                    &bias_beta,
                                    output_desc, output.data_ptr<float>()));
    }

    // Free workspace if allocated
    if (workspace != nullptr) {
        CHECK_CUDA(cudaFree(workspace));
    }

    // Destroy descriptors
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));

    return output;
}

// ===================== conv1d_forward =====================
torch::Tensor conv1d_forward(torch::Tensor input, torch::Tensor weight,
                             c10::optional<torch::Tensor> bias, int64_t stride,
                             int64_t padding) {
    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels, kernel_size)");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input dtype must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight dtype must be float32");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.has_value() && bias.value().defined()) {
        TORCH_CHECK(bias.value().is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.value().dim() == 1, "bias must be 1D");
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "bias dtype must be float32");
        bias_contig = bias.value().contiguous();
    }

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(0);
    int64_t kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(1) == in_channels, "weight second dimension must match input channels");

    int64_t output_length = (length + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(output_length > 0, "output length must be positive");

    auto output = torch::empty({batch, out_channels, output_length}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUDNN(cudnnCreate(&handle));
    }
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    int dims[4] = {static_cast<int>(batch), static_cast<int>(in_channels), 1, static_cast<int>(length)};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           dims[0], dims[1], dims[2], dims[3]));

    int out_dims[4] = {static_cast<int>(batch), static_cast<int>(out_channels), 1, static_cast<int>(output_length)};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           out_dims[0], out_dims[1], out_dims[2], out_dims[3]));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, static_cast<int>(out_channels), 1, 1));

    // Filter layout: [out_channels, in_channels, 1, kernel_size]
    int filter_dims[4] = {static_cast<int>(out_channels), static_cast<int>(in_channels), 1, static_cast<int>(kernel_size)};
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding, padding,
                                                stride, stride,
                                                1, 1,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                                        input_desc,
                                                        filter_desc,
                                                        conv_desc,
                                                        output_desc,
                                                        algo,
                                                        &workspace_size));

    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(handle,
                                        &alpha,
                                        input_desc, input_contig.data_ptr<float>(),
                                        filter_desc, weight_contig.data_ptr<float>(),
                                        conv_desc,
                                        algo,
                                        workspace, workspace_size,
                                        &beta,
                                        output_desc, output.data_ptr<float>()));

    if (bias_contig.defined()) {
        float bias_alpha = 1.0f, bias_beta = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(handle,
                                    &bias_alpha,
                                    bias_desc, bias_contig.data_ptr<float>(),
                                    &bias_beta,
                                    output_desc, output.data_ptr<float>()));
    }

    if (workspace != nullptr) {
        CHECK_CUDA(cudaFree(workspace));
    }

    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));

    return output;
}

// ===================== sum_dim_forward =====================
torch::Tensor sum_dim_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input dtype must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");

    // Adjust negative dim
    int64_t dim_normalized = dim < 0 ? dim + input.dim() : dim;
    TORCH_CHECK(dim_normalized >= 0 && dim_normalized < input.dim(), "dim out of range");

    // For this model, we only support sum over dim=1, but we can generalize later.
    TORCH_CHECK(dim_normalized == 1, "sum_dim_forward currently only supports dim=1");

    int64_t batch = input.size(0);
    int64_t channels = input.size(1);
    int64_t length = input.size(2);

    // Allocate output [batch, length]
    auto output = torch::empty({batch, length}, input.options());

    // Get strides (in elements)
    long long in_stride_batch = input.stride(0);
    long long in_stride_channel = input.stride(1);
    long long in_stride_length = input.stride(2);
    long long out_stride_batch = output.stride(0);
    long long out_stride_length = output.stride(1);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher from .cu
    sum_dim_launcher(output.data_ptr<float>(), input.data_ptr<float>(),
                     1,  // dim
                     static_cast<int>(batch), static_cast<int>(channels), static_cast<int>(length),
                     in_stride_batch, in_stride_channel, in_stride_length,
                     out_stride_batch, out_stride_length,
                     stream);

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA kernel launch failed");

    return output;
}

// ===================== Register bindings =====================
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "1D transposed convolution using cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("conv1d_forward", &conv1d_forward,
          "1D convolution using cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("sum_dim_forward", &sum_dim_forward,
          "Sum over dimension (currently only dim=1)",
          pybind11::arg("input"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
