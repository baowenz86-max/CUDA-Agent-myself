#include <torch/types.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>

// Helper to get current CUDA stream
static cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// conv_transpose1d_forward implementation using cuDNN
torch::Tensor conv_transpose1d_forward(torch::Tensor x,
                                       torch::Tensor weight,
                                       torch::Tensor bias,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t output_padding,
                                       int64_t groups = 1,
                                       int64_t dilation = 1) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [N,C,L]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [C_in,C_out,kernel_size]");

    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    }

    int64_t N = x.size(0);
    int64_t C_in = x.size(1);
    int64_t L_in = x.size(2);
    int64_t C_out = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "weight shape mismatch");
    TORCH_CHECK(groups == 1, "only groups=1 supported");
    TORCH_CHECK(dilation == 1, "only dilation=1 supported");

    int64_t L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    // Ensure contiguous tensors for cuDNN
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.defined()) {
        bias_contig = bias.contiguous();
    }

    auto output = torch::empty({N, C_out, L_out}, x.options());

    cudaStream_t stream = getCurrentCUDAStream();

    // cuDNN handle
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, stream);

    // Tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors: NCHW format, but for 1D we treat as N,C,L,1
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, 1, L_in);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, 1, L_out);
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1);

    // Filter descriptor: weight shape [C_in, C_out, kernel_size] treated as [C_out, C_in/groups, kernel_size]
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, 1, kernel_size);

    // Convolution descriptor: use cross-correlation with transpose
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    padding, padding,
                                    stride, stride,
                                    dilation, dilation,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH);

    float alpha = 1.0f, beta = 0.0f;

    // Compute transposed convolution via backward data
    cudnnConvolutionBackwardData(cudnn_handle,
                                 &alpha,
                                 w_desc, weight_contig.data_ptr<float>(),
                                 x_desc, x_contig.data_ptr<float>(),
                                 conv_desc,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                 nullptr, 0,
                                 &beta,
                                 y_desc, output.data_ptr<float>());

    // Add bias if present
    if (bias_contig.defined()) {
        cudnnAddTensor(cudnn_handle, &alpha, b_desc, bias_contig.data_ptr<float>(),
                       &alpha, y_desc, output.data_ptr<float>());
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}
