#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Helper to check cuDNN status
#define CHECK_CUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
        TORCH_CHECK(false, "cuDNN error: ", cudnnGetErrorString(status)); \
    } \
}

// PyTorch wrapper for ConvTranspose1d using cuDNN
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int stride,
    int padding,
    int output_padding
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (batch, in_channels, length)");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (in_channels, out_channels, kernel_size)");

    // Handle bias
    torch::Tensor bias;
    if (bias_opt.has_value() && bias_opt.value().defined()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");
        TORCH_CHECK(bias.size(0) == weight.size(1), "Bias size must match out_channels");
    }

    // Ensure input, weight, and bias are contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    // Dimensions
    int batch = input_contig.size(0);
    int in_channels = input_contig.size(1);
    int length_in = input_contig.size(2);
    int out_channels = weight_contig.size(1);
    int kernel_size = weight_contig.size(2);

    // Compute output length using PyTorch formula
    int length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(length_out > 0, "Output length must be positive");

    // Allocate output tensor
    auto output = torch::empty({batch, out_channels, length_out}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle (create per call for simplicity; could reuse but okay)
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // Set tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set input descriptor: [batch, in_channels, length] -> treat as [batch, in_channels, length, 1]? 
    // cuDNN supports NCHW for 2D, but for 1D we can use NCHW with height=1? Actually cuDNN has 2D and 3D only.
    // For 1D conv, PyTorch internally uses 2D with height=1. Alternatively, use CUDNN_TENSOR_NCHW with dims [batch, in_channels, length].
    // cuDNN supports rank >= 4? Actually, cudnnSetTensor4dDescriptor can be used if we treat 1D as 2D with height=1.
    // However, simpler: since this is a small model, we can implement using cudnnConvolutionBackwardData or cudnnConvolutionForward with specific settings.
    // Let's set input as [batch, in_channels, 1, length] and output as [batch, out_channels, 1, length_out] (2D spatial).
    // Then use cudnnConvolutionBackwardData with appropriate filter and conv descriptors.
    
    // Use 2D descriptors with height=1
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, in_channels, 1, length_in));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, out_channels, 1, length_out));
    if (bias_contig.defined()) {
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, out_channels, 1, 1));
    }

    // Set filter descriptor: [in_channels, out_channels, kernel_size] -> for backward data, filter is [in_channels, out_channels, kernel_size]
    // cuDNN Filter descriptor for conv: [out_channels, in_channels, kernel_height, kernel_width] but for backward data, weight layout is [in_channels, out_channels, kh, kw]
    // We'll set filter as [in_channels, out_channels, kernel_size, 1]
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           in_channels, out_channels, kernel_size, 1));

    // Set convolution descriptor for backward data (which is equivalent to transposed convolution)
    // For ConvTranspose, we use cudnnConvolutionBackwardData with mode CUDNN_CROSS_CORRELATION.
    // Set conv descriptor with appropriate padding, stride, dilation=1.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                 padding, padding,   // pad_h, pad_w (equal)
                                                 stride, stride,     // stride_h, stride_w (equal)
                                                 1, 1,               // dilation_h, dilation_w
                                                 CUDNN_CROSS_CORRELATION,
                                                 CUDNN_DATA_FLOAT));

    // Determine workspace size for backward data algorithm
    cudnnConvolutionBwdDataAlgo_t algo;
    // Query best algo
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        filter_desc,
        output_desc,   // dy (output gradient) -> for backward data, dy has shape of original conv output
        conv_desc,
        input_desc,    // dx (input gradient) -> shape of original conv input
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo));

    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        algo,
        &workspace_size));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Note: In ConvTranspose, the roles of input and output are reversed relative to forward conv.
    // In cudnnConvolutionBackwardData, dx is computed from dy, w, and conv_desc, where:
    //   dy shape = original conv output shape (here: output of transposed conv? Actually careful).
    // For transposed conv, PyTorch calls cudnnConvolutionBackwardData with filter, diff_output (the input to transposed conv?), etc.
    // We can map:
    //   dy = input_contig (shape [batch, in_channels, length_in]) ??? No, backward data uses dy as the "output gradient" which is gradient w.r.t. conv forward output, which for transposed conv is the input x? Let's derive.
    // Better: Use the approach of swapping input and output descriptors for backward data.
    // For transposed convolution (deconvolution), the operation can be implemented as the gradient of a standard convolution with respect to its input, using the same weight but with the roles of the stride and padding interpreted accordingly.
    // Specifically, torch.nn.functional.conv_transpose1d(input, weight, bias, stride, padding, output_padding) is equivalent to the gradient of a convolution with input size = output size, output size = input size, but with weight transpose? Actually, the transposed convolution can be computed using cudnnConvolutionBackwardData with:
    //   w = weight (shape [in_channels, out_channels, kernel_size])
    //   dy = grad_output w.r.t. the transposed conv output, which is actually the input tensor x of shape [batch, in_channels, length_in]
    //   dx = grad_input w.r.t. the transposed conv input, which is the output tensor of shape [batch, out_channels, length_out]
    // Wait, this seems reversed. Let's check: In standard convolution, forward: y = conv(x, w). Backward data: dx = conv_backward_data(dy, w). If we apply backward data to produce x from y using w, that's exactly a transposed convolution? Actually, transposed convolution is the gradient of the conv with respect to the input, i.e., if we have a conv layer with weight w and stride s and padding p, then its backward pass computes dx from dy using a transposed convolution with same w and stride s and padding p, but with output_padding to handle ambiguity. So to implement transposed convolution as a cuDNN call, we can use cudnnConvolutionBackwardData with:
    //   dy = input (our original x to conv_transpose, shape [batch, in_channels, length_in])
    //   w = weight (shape [in_channels, out_channels, kernel_size])
    //   dx = output (shape [batch, out_channels, length_out])
    //   conv_desc must describe the *original* convolution that we are differentiating. That original convolution should have input shape = output of transposed conv, output shape = input of transposed conv, weight shape = [in_channels, out_channels, kernel_size], stride = stride, padding = padding, dilation = 1. But the original convolution's input channels = out_channels, output channels = in_channels. However, weight shape is [in_channels, out_channels, kernel_size], which is the same as for the original convolution? In cuDNN, for backward data, the filter descriptor still has dimensions [K, C, R, S] where K = number of output feature maps of the original convolution (= in_channels), C = number of input feature maps of the original convolution (= out_channels). So filter desc should be [in_channels, out_channels, kernel_size, 1]. That matches our weight shape.
    //   dx (gradient w.r.t. original input) will have shape [batch, out_channels, length_out].
    // So we should set:
    //   dy_desc (input gradient? Actually in cudnnConvolutionBackwardData, the arguments are:
    //      handle, filterDesc, dyDesc, convDesc, dxDesc, algo, workspace, workspaceSize, beta, dx).
    //   dyDesc: descriptor for the gradient of the loss w.r.t. the output of the original convolution, which has shape [batch, out_channels? Wait, original convolution output channels = in_channels, so dyDesc should be [batch, in_channels, length_in] and dxDesc [batch, out_channels, length_out].
    // So we set dyDesc = input_contig description (since our input x is treated as dy), dxDesc = output description.
    // But our current setup above used input_desc for dx and output_desc for dy, which is reversed. Let's correct.
    // We need to reorder the descriptions:
    //   dxDesc = output tensor descriptor (shape [batch, out_channels, length_out])
    //   dyDesc = input tensor descriptor (shape [batch, in_channels, length_in])
    //   filterDesc = weight descriptor (shape [in_channels, out_channels, kernel_size])
    //   convDesc describes the original convolution with stride/padding/dilation.
    // Then call cudnnConvolutionBackwardData(handle, &alpha, filterDesc, weight, dyDesc, input, convDesc, algo, workspace, workspace_size, &beta, dxDesc, output).
    // Note: alpha=1, beta=0 initially; beta may be used if bias is added separately, but we can incorporate bias via beta or separate addition. Since output is zero-initialized, we can set beta=0 and then add bias using cudnnAddTensor or a custom kernel. Alternatively, use beta=1 if we pre-fill output with bias. But we don't have a pre-filled output. So after computing dx, we add bias using cudnnAddTensor.

    // Correct descriptor assignments:
    // input_desc (dy) should be [batch, in_channels, 1, length_in]
    // output_desc (dx) should be [batch, out_channels, 1, length_out]
    // We already set them accordingly, but we had assigned input_desc to input_contig and output_desc to output. That's correct.
    // So dyDesc = input_desc, dxDesc = output_desc.

    // Now call backward data
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        weight_contig.data_ptr<float>(),
        input_desc,
        input_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        output_desc,
        output.data_ptr<float>()));

    // Add bias if defined
    if (bias_contig.defined()) {
        float beta_bias = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                                   &alpha,
                                   bias_desc,
                                   bias_contig.data_ptr<float>(),
                                   &beta_bias,
                                   output_desc,
                                   output.data_ptr<float>()));
    }

    // Free workspace
    if (workspace) {
        cudaFree(workspace);
    }

    // Destroy descriptors and handle
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "conv_transpose1d_forward failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_conv_transpose1d(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("stride") = 1,
          py::arg("padding") = 0,
          py::arg("output_padding") = 0);
}

REGISTER_BINDING(conv_transpose1d, register_conv_transpose1d);
