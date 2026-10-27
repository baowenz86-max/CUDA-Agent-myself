// Binding for custom ops: linspace_1d, elementwise_sub, conv_transpose1d
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" cudaError_t linspace_1d_launcher(
    float* output, float start, float end, int steps, cudaStream_t stream);

extern "C" cudaError_t elementwise_sub_launcher(
    const float* x, const float* y, float* out,
    int batch, int channels, int length,
    int x_batch_stride, int x_channel_stride, int y_stride,
    cudaStream_t stream);

// Helper to get cuDNN error string
static const char* cudnnGetErrorString(cudnnStatus_t status) {
    switch (status) {
        case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR";
        case CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING: return "CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING";
        case CUDNN_STATUS_RUNTIME_IN_PROGRESS: return "CUDNN_STATUS_RUNTIME_IN_PROGRESS";
        case CUDNN_STATUS_RUNTIME_FP_OVERFLOW: return "CUDNN_STATUS_RUNTIME_FP_OVERFLOW";
        default: return "Unknown cudnn status";
    }
}

// Helper to check CUDA errors
static void checkCudaError(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// Helper to check cuDNN errors
static void checkCudnnError(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

// Helper to get current CUDA stream
static cudaStream_t getCurrentCudaStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// linspace_1d binding
torch::Tensor linspace_1d_forward(double start, double end, int64_t steps, int64_t device_index) {
    TORCH_CHECK(steps >= 0, "steps must be non-negative, got ", steps);
    TORCH_CHECK(device_index >= 0, "device_index must be non-negative, got ", device_index);

    // Create tensor on specified device using empty_like? No, we need a new tensor. Use torch::empty with explicit options.
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, device_index));
    auto output = torch::empty({steps}, options);

    if (steps == 0) {
        return output;
    }

    cudaStream_t stream = getCurrentCudaStream();
    cudaError_t err = linspace_1d_launcher(
        output.data_ptr<float>(),
        static_cast<float>(start),
        static_cast<float>(end),
        static_cast<int>(steps),
        stream
    );
    checkCudaError(err, "linspace_1d_launcher failed");
    return output;
}

// elementwise_sub binding
torch::Tensor elementwise_sub_forward(torch::Tensor x, torch::Tensor y) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(y.is_cuda(), "y must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(y.dtype() == torch::kFloat32, "y must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(y.dim() == 1, "y must be 1D (length)");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t length = x.size(2);
    TORCH_CHECK(y.size(0) == length, "y length (", y.size(0), ") must match x last dim (", length, ")");

    // Ensure contiguous for kernel; if not, we can use strides but kernel supports strides
    // For output, allocate contiguous tensor of same shape
    auto output = torch::empty_like(x); // preserves shape, dtype, device, but will be contiguous by default

    cudaStream_t stream = getCurrentCudaStream();

    // Get strides in elements
    int64_t x_batch_stride = x.stride(0);
    int64_t x_channel_stride = x.stride(1);
    int64_t y_stride = y.stride(0);

    cudaError_t err = elementwise_sub_launcher(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(channels),
        static_cast<int>(length),
        static_cast<int>(x_batch_stride),
        static_cast<int>(x_channel_stride),
        static_cast<int>(y_stride),
        stream
    );
    checkCudaError(err, "elementwise_sub_launcher failed");
    return output;
}

// conv_transpose1d binding using cuDNN
torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor or None");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32 if provided");
    }
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    if (bias.defined()) {
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");
    }

    int64_t batch_size = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t input_length = x.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    // Validate weight shape matches in_channels
    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels must match x in_channels");
    if (bias.defined()) {
        TORCH_CHECK(bias.size(0) == out_channels, "bias size must match out_channels");
    }

    // Compute output length
    int64_t output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "computed output length must be positive, got ", output_length);

    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto output = torch::empty({batch_size, out_channels, output_length}, options);

    // If output is empty (batch or channels zero), return empty
    if (batch_size == 0 || out_channels == 0 || output_length == 0) {
        return output;
    }

    // Get current stream
    cudaStream_t stream = getCurrentCudaStream();

    // Create cuDNN handle for this stream
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    checkCudnnError(status, "cudnnCreate failed");
    status = cudnnSetStream(cudnn_handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnSetStream failed");
    }

    // Set up tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    status = cudnnCreateTensorDescriptor(&x_desc);
    if (status == CUDNN_STATUS_SUCCESS) {
        status = cudnnCreateTensorDescriptor(&y_desc);
    }
    if (status == CUDNN_STATUS_SUCCESS) {
        status = cudnnCreateTensorDescriptor(&b_desc);
    }
    if (status != CUDNN_STATUS_SUCCESS) {
        // clean up partially created descriptors
        if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
        if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
        if (b_desc) cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnCreateTensorDescriptor failed");
    }

    // Set x descriptor: [batch_size, in_channels, input_length] NCHW format for 1D
    status = cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,
        in_channels,
        input_length,
        1  // height dimension (unused)
    );

    if (status == CUDNN_STATUS_SUCCESS) {
        status = cudnnSetTensor4dDescriptor(
            y_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            batch_size,
            out_channels,
            output_length,
            1
        );
    }

    if (status == CUDNN_STATUS_SUCCESS && bias.defined()) {
        // Bias descriptor for 1D: can use 4D with dimensions [1, out_channels, 1, 1]
        status = cudnnSetTensor4dDescriptor(
            b_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            out_channels,
            1,
            1
        );
    }

    if (status != CUDNN_STATUS_SUCCESS) {
        // clean up and throw
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnSetTensor4dDescriptor failed");
    }

    // Set up filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    status = cudnnCreateFilterDescriptor(&filter_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnCreateFilterDescriptor failed");
    }

    // Reorder weight from [in_channels, out_channels, kernel_size] to [out_channels, in_channels, kernel_size]
    // We'll do a host-side reorder because weight is small.
    std::vector<float> weight_host(in_channels * out_channels * kernel_size);
    cudaError_t cuda_err = cudaMemcpyAsync(
        weight_host.data(),
        weight.data_ptr<float>(),
        weight_host.size() * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream
    );
    checkCudaError(cuda_err, "cudaMemcpyAsync weight to host failed");
    cuda_err = cudaStreamSynchronize(stream);
    checkCudaError(cuda_err, "cudaStreamSynchronize failed");

    std::vector<float> weight_reordered(in_channels * out_channels * kernel_size);
    for (int64_t oc = 0; oc < out_channels; ++oc) {
        for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = (ic * out_channels + oc) * kernel_size + k;
                int64_t dst_idx = (oc * in_channels + ic) * kernel_size + k;
                weight_reordered[dst_idx] = weight_host[src_idx];
            }
        }
    }

    float* d_weight_reordered = nullptr;
    cuda_err = cudaMalloc(&d_weight_reordered, weight_reordered.size() * sizeof(float));
    checkCudaError(cuda_err, "cudaMalloc for reordered weight failed");

    cuda_err = cudaMemcpyAsync(
        d_weight_reordered,
        weight_reordered.data(),
        weight_reordered.size() * sizeof(float),
        cudaMemcpyHostToDevice,
        stream
    );
    checkCudaError(cuda_err, "cudaMemcpyAsync reordered weight to device failed");

    // Set filter descriptor with reordered weight.
    status = cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        out_channels,
        in_channels,
        kernel_size,
        1
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(d_weight_reordered);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnSetFilter4dDescriptor failed");
    }

    // Set convolution descriptor for backward data (transposed conv)
    cudnnConvolutionDescriptor_t conv_desc;
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(d_weight_reordered);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnCreateConvolutionDescriptor failed");
    }

    // Set convolution descriptor: 1D convolution, padding, stride
    status = cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding,        // vertical padding
        0,              // horizontal padding
        stride,         // vertical stride
        1,              // horizontal stride
        1,              // dilation vertical
        1,              // dilation horizontal
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(d_weight_reordered);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnSetConvolution2dDescriptor failed");
    }

    // Find best algorithm for backward data
    cudnnConvolutionBwdDataAlgo_t algo;
    status = cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        filter_desc,
        y_desc,   // dxDesc
        conv_desc,
        x_desc,   // dyDesc
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(d_weight_reordered);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnGetConvolutionBackwardDataAlgorithm failed");
    }

    // Allocate workspace
    size_t workspace_size = 0;
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        filter_desc,
        x_desc,  // dyDesc
        conv_desc,
        y_desc,  // dxDesc
        algo,
        &workspace_size
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(d_weight_reordered);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
    }

    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        cuda_err = cudaMalloc(&d_workspace, workspace_size);
        if (cuda_err != cudaSuccess) {
            cudaFree(d_weight_reordered);
            cudnnDestroyTensorDescriptor(x_desc);
            cudnnDestroyTensorDescriptor(y_desc);
            cudnnDestroyTensorDescriptor(b_desc);
            cudnnDestroyFilterDescriptor(filter_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroy(cudnn_handle);
            checkCudaError(cuda_err, "cudaMalloc workspace failed");
        }
    }

    // Execute convolution backward data
    float alpha = 1.0f;
    float beta = 0.0f;
    status = cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        d_weight_reordered,
        x_desc,  // dyDesc = input
        x.data_ptr<float>(),
        conv_desc,
        algo,
        d_workspace,
        workspace_size,
        &beta,
        y_desc,  // dxDesc = output
        output.data_ptr<float>()
    );

    if (status != CUDNN_STATUS_SUCCESS) {
        if (d_workspace) cudaFree(d_workspace);
        cudaFree(d_weight_reordered);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn_handle);
        checkCudnnError(status, "cudnnConvolutionBackwardData failed");
    }

    // Add bias if provided
    if (bias.defined()) {
        float alpha_bias = 1.0f;
        float beta_bias = 1.0f;
        status = cudnnAddTensor(
            cudnn_handle,
            &alpha_bias,
            b_desc,
            bias.data_ptr<float>(),
            &beta_bias,
            y_desc,
            output.data_ptr<float>()
        );
        if (status != CUDNN_STATUS_SUCCESS) {
            if (d_workspace) cudaFree(d_workspace);
            cudaFree(d_weight_reordered);
            cudnnDestroyTensorDescriptor(x_desc);
            cudnnDestroyTensorDescriptor(y_desc);
            cudnnDestroyTensorDescriptor(b_desc);
            cudnnDestroyFilterDescriptor(filter_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            cudnnDestroy(cudnn_handle);
            checkCudnnError(status, "cudnnAddTensor bias failed");
        }
    }

    // Cleanup
    if (d_workspace) cudaFree(d_workspace);
    cudaFree(d_weight_reordered);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("linspace_1d", &linspace_1d_forward,
          "Generate linear space on CUDA",
          pybind11::arg("start"),
          pybind11::arg("end"),
          pybind11::arg("steps"),
          pybind11::arg("device"));
    m.def("elementwise_sub", &elementwise_sub_forward,
          "Elementwise subtraction with broadcasting along last dimension",
          pybind11::arg("x"),
          pybind11::arg("y"));
    m.def("conv_transpose1d", &conv_transpose1d_forward,
          "1D transposed convolution using cuDNN",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
