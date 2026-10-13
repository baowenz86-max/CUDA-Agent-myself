#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void bias_add_forward_launcher(
    float* out,
    const float* in,
    const float* bias,
    int batch,
    int channels,
    int length,
    int batch_stride,
    int channel_stride,
    int inner_stride,
    cudaStream_t stream);

extern "C" void round_elementwise_forward_launcher(
    float* out,
    const float* in,
    int ndims,
    const int* shape,
    const int* strides,
    int total_elements,
    cudaStream_t stream);

// Helper to check cuDNN status
#define CHECK_CUDNN(expr)                                                      \
    do {                                                                       \
        cudnnStatus_t status = (expr);                                         \
        if (status != CUDNN_STATUS_SUCCESS) {                                  \
            throw std::runtime_error(cudnnGetErrorString(status));            \
        }                                                                      \
    } while (0)

// ============ ConvTranspose1d forward using cuDNN ============
torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    int64_t batch = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t length_in = x.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(output_padding < stride, "output_padding must be less than stride");

    // Compute output length
    int64_t length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(length_out > 0, "output length must be positive");

    // Allocate output tensor (contiguous, same device and dtype as input)
    auto out = torch::empty({batch, out_channels, length_out}, x.options());

    // Ensure input and weight are contiguous for cuDNN
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors: For 1D convolution, cuDNN expects 3D tensors with N,C,W but uses layout NCHW for 2D.
    // For 1D, we can use CUDNN_TENSOR_NCHW with dimensions [batch, in_channels, 1, length_in] and treat it as 2D with height=1.
    // Alternatively use cudnnSetTensor4dDescriptor with dims [batch, in_channels, 1, length_in].
    // We will use 4D descriptors with height dimension = 1.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)batch,
        (int)in_channels,
        1,
        (int)length_in));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)batch,
        (int)out_channels,
        1,
        (int)length_out));

    // For ConvTranspose1d, weight shape is [in_channels, out_channels, kernel_size].
    // cuDNN filter descriptor expects [out_channels, in_channels, filter_h, filter_w] for convolution.
    // For transposed convolution (backward data), weight is interpreted as [out_channels, in_channels, kernel_h, kernel_w].
    // PyTorch stores weight as [in_channels, out_channels, kernel_size], so we need to transpose dimensions or use a custom filter descriptor.
    // We'll create a filter descriptor with shape [out_channels, in_channels, 1, kernel_size] and pass the weight with adjusted layout.
    // However, the weight memory layout as stored is [in_channels, out_channels, kernel_size]; cuDNN expects [out_channels, in_channels, 1, kernel_size] contiguous.
    // We can either create a transposed copy or set the filter descriptor with strides to interpret the memory correctly.
    // Since weight_contig is contiguous, we can use cudnnSetFilter4dDescriptor with CUDNN_TENSOR_NCHW and dims [out_channels, in_channels, 1, kernel_size],
    // but the memory layout is not as expected. Instead, we can use cudnnSetFilter4dDescriptor with CUDNN_TENSOR_NCHW and dims [in_channels, out_channels, 1, kernel_size]
    // and use CUDNN_CONVOLUTION_MODE_CROSS_CORRELATION; but PyTorch uses cross-correlation. We need to investigate cuDNN transposed convolution API.
    // Actually cuDNN transposed convolution is performed by calling cudnnConvolutionBackwardData, where the filter descriptor has the same layout as the forward convolution weight.
    // In forward convolution, weight shape is [out_channels, in_channels, filter_h, filter_w]. For transposed conv, we can use the same weight layout by swapping in/out channels:
    // if we set input descriptor with out_channels (as output of transposed), output descriptor with in_channels (as input of transposed), and filter descriptor as [in_channels, out_channels, 1, kernel_size], then the weight memory layout [in_channels, out_channels, kernel_size] matches exactly.
    // So we can call cudnnConvolutionBackwardData with:
    //   dy_desc: y_desc (shape [batch, out_channels, 1, length_out])
    //   w_desc: filter descriptor with dims [in_channels, out_channels, 1, kernel_size]
    //   conv_desc: convolution descriptor with stride, padding, etc.
    //   dx_desc: x_desc (shape [batch, in_channels, 1, length_in])
    //   dx: input tensor (x)
    //   w: weight tensor
    //   dy: output tensor (y)
    // This is the standard approach for transposed convolution (backward data of a convolution whose forward has input=dy, output=dx).
    // Let's implement that.

    // Set filter descriptor for [in_channels, out_channels, 1, kernel_size] (as needed for backward data with swapped roles)
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        (int)in_channels,
        (int)out_channels,
        1,
        (int)kernel_size));

    // Set convolution descriptor: for backward data, stride and padding are as in forward conv, but we need to set them appropriately.
    // In forward conv, output size = (input + 2*padding - kernel)/stride + 1.
    // In backward data, we want to compute dx given dy and w, so we set convolution descriptor with stride, padding, and dilation.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        (int)padding,  // pad_h
        0,             // pad_w (1D)
        (int)stride,   // stride_h
        1,             // stride_w
        1, 1,          // dilation_h, dilation_w
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT));

    // Note: output_padding is not directly supported by cuDNN; it is handled by ensuring the output size is correct.
    // The length_out already includes output_padding, so the descriptor y_desc has the correct size.

    // Determine algorithm
    cudnnConvolutionBwdDataAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        w_desc,
        y_desc,
        conv_desc,
        x_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo));

    // Allocate workspace
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        w_desc,
        y_desc,
        conv_desc,
        x_desc,
        algo,
        &workspace_size));

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;

    // Execute backward data: compute x (dx) from y (dy) and weight
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        w_desc,
        weight_contig.data_ptr<float>(),
        y_desc,
        out.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        x_desc,
        x_contig.data_ptr<float>()));

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return out;
}

// ============ Bias add forward ============
torch::Tensor bias_add_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t length = x.size(2);
    TORCH_CHECK(bias.size(0) == channels, "bias size must match channels");

    auto bias_contig = bias.contiguous();
    auto out = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int64_t batch_stride = x.stride(0);
    int64_t channel_stride = x.stride(1);
    int64_t inner_stride = x.stride(2);

    int total_elements = batch * channels * length;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bias_add_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        (int)batch,
        (int)channels,
        (int)length,
        (int)batch_stride,
        (int)channel_stride,
        (int)inner_stride,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bias_add_forward kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// ============ Round elementwise forward ============
torch::Tensor round_elementwise_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto out = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int ndims = x.dim();
    int total_elements = x.numel();

    std::vector<int> shape_cpu(ndims);
    std::vector<int> strides_cpu(ndims);
    for (int i = 0; i < ndims; ++i) {
        shape_cpu[i] = (int)x.size(i);
        strides_cpu[i] = (int)x.stride(i);
    }

    int* shape_dev = nullptr;
    int* strides_dev = nullptr;
    cudaMalloc(&shape_dev, ndims * sizeof(int));
    cudaMalloc(&strides_dev, ndims * sizeof(int));
    cudaMemcpyAsync(shape_dev, shape_cpu.data(), ndims * sizeof(int), cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream());
    cudaMemcpyAsync(strides_dev, strides_cpu.data(), ndims * sizeof(int), cudaMemcpyHostToDevice, c10::cuda::getCurrentCUDAStream());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    round_elementwise_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        ndims,
        shape_dev,
        strides_dev,
        total_elements,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "round_elementwise_forward kernel launch failed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaFree(shape_dev);
    cudaFree(strides_dev);

    return out;
}

// ============ Registration ============
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "1D transposed convolution without bias",
          py::arg("x"), py::arg("weight"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("bias_add_forward", &bias_add_forward,
          "Add bias to each channel of a 3D tensor",
          py::arg("x"), py::arg("bias"));
    m.def("round_elementwise_forward", &round_elementwise_forward,
          "Round elements to nearest integer, half to even",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
