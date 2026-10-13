#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>

#include <vector>

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

// ============ ConvTranspose1d forward (cuDNN) ============
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

    // Allocate output tensor (contiguous)
    auto out = torch::empty({batch, out_channels, length_out}, x.options());

    // Ensure weight is contiguous
    auto weight_contig = weight.contiguous();
    // Ensure input is contiguous for cuDNN; if not, make contiguous
    auto x_contig = x.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle (static or per-call)
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreate failed");
    status = cudnnSetStream(cudnn_handle, stream);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetStream failed");

    // Set up tensor descriptors for input, output, weight (filter)
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Input tensor: [batch, in_channels, length] -> cuDNN expects NCHW, but for 1D use N, C, W
    int dims_x[3] = {(int)batch, (int)in_channels, (int)length_in};
    int strides_x[3] = {(int)(in_channels * length_in), (int)length_in, 1};
    cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dims_x, strides_x);

    // Output tensor: [batch, out_channels, length_out]
    int dims_y[3] = {(int)batch, (int)out_channels, (int)length_out};
    int strides_y[3] = {(int)(out_channels * length_out), (int)length_out, 1};
    cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 3, dims_y, strides_y);

    // Weight (filter) descriptor: [in_channels, out_channels, kernel_size]
    // cuDNN expects K, C, R for convolution, but for transposed convolution,
    // the weight layout is [in_channels, out_channels, kernel_size].
    int filter_dims[3] = {(int)in_channels, (int)out_channels, (int)kernel_size};
    cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, filter_dims);

    // Convolution descriptor for transposed convolution (backward data)
    // We need to set up padding, stride, dilation (default 1), and output padding implicitly.
    int conv_pad[1] = {(int)padding};
    int conv_stride[1] = {(int)stride};
    int conv_dilation[1] = {1};
    cudnnSetConvolutionNdDescriptor(
        conv_desc,
        1, // array length for 1D
        conv_pad,
        conv_stride,
        conv_dilation,
        CUDNN_CONVOLUTION,
        CUDNN_DATA_FLOAT);

    // Transposed convolution: use cudnnConvolutionBackwardData
    // Forward of transposed conv is equivalent to backward data of regular conv with the same weight layout.
    float alpha = 1.0f, beta = 0.0f;

    // Find algorithm
    cudnnConvolutionBwdDataAlgo_t algo;
    status = cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        w_desc,
        y_desc,
        conv_desc,
        x_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionBackwardDataAlgorithm failed");

    // Workspace size
    size_t workspace_size = 0;
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        w_desc,
        y_desc,
        conv_desc,
        x_desc,
        algo,
        &workspace_size);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionBackwardDataWorkspaceSize failed");

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Execute backward data: output = convTranspose(input) with weight
    // Arguments: handle, alpha, w_desc, weight, y_desc, dy (input to backward data), conv_desc, algo, workspace, workspace_size, beta, x_desc, dx (output)
    status = cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        w_desc,
        weight_contig.data_ptr<float>(),
        y_desc,
        x_contig.data_ptr<float>(), // x_contig is dy
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        x_desc,
        out.data_ptr<float>()); // out is dx

    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionBackwardData failed");

    if (workspace) {
        cudaFree(workspace);
    }

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
