#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include "../binding_registry.h"

// Helper to calculate output length for ConvTranspose1d
torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must have shape [N, C_in, L_in]");
    TORCH_CHECK(weight.dim() == 3, "weight must have shape [C_in, C_out, kernel_size]");
    
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "bias must have shape [C_out]");
        TORCH_CHECK(bias.size(0) == weight.size(1), "bias size must match C_out");
    }
    
    // Make inputs contiguous
    x = x.contiguous();
    weight = weight.contiguous();
    if (bias.defined()) {
        bias = bias.contiguous();
    }
    
    // Extract dimensions
    int64_t N = x.size(0);          // batch size
    int64_t C_in = x.size(1);       // input channels
    int64_t L_in = x.size(2);       // input length
    int64_t C_out = weight.size(1); // output channels
    int64_t K = weight.size(2);     // kernel size
    
    // Validate weight shape
    TORCH_CHECK(weight.size(0) == C_in, "weight shape[0] must equal C_in");
    
    // Calculate output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "calculated output length must be positive");
    
    // Allocate output tensor
    auto output = torch::empty({N, C_out, L_out}, x.options());
    
    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    // Initialize cuDNN
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreate failed");
    cudnnSetStream(cudnn_handle, stream);
    
    // Create tensor descriptor for input (x)
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               N, C_in, L_in, 1);
    
    // Create tensor descriptor for output (y)
    cudnnTensorDescriptor_t y_desc;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc,
                               CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT,
                               N, C_out, L_out, 1);
    
    // Create filter descriptor for weight
    // Note: For transposed convolution, cuDNN expects weight shape [C_out, C_in, K]
    // but PyTorch ConvTranspose1d weight is [C_in, C_out, K].
    // We need to treat as cudnnConvolutionBwdData with filter descriptor.
    cudnnFilterDescriptor_t w_desc;
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnSetFilter4dDescriptor(w_desc,
                               CUDNN_DATA_FLOAT,
                               CUDNN_TENSOR_NCHW,
                               C_in, C_out, K, 1);
    
    // Transposed convolution is equivalent to convolution backward data.
    // We'll use cudnnConvolutionBwdData which does not require bias in desc.
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    padding, padding, // vertical, horizontal pad
                                    stride, stride,   // vertical, horizontal stride
                                    1, 1,             // dilation
                                    CUDNN_CROSS_CORRELATION, // PyTorch uses cross-correlation
                                    CUDNN_DATA_FLOAT);
    
    // Determine algorithms
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    status = cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle,
        w_desc,
        y_desc,
        conv_desc,
        x_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &bwd_data_algo);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionBackwardDataAlgorithm failed");
    
    // Get workspace size
    size_t workspace_size = 0;
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        w_desc,
        y_desc,
        conv_desc,
        x_desc,
        bwd_data_algo,
        &workspace_size);
    if (status != CUDNN_STATUS_SUCCESS) {
        // Some algorithms may not have workspace query; fallback to deterministic
        workspace_size = 0;
    }
    
    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    // Execute convolution backward data (transposed convolution)
    float alpha = 1.0f;
    float beta = 0.0f;
    status = cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        w_desc, weight.data_ptr<float>(),
        y_desc, output.data_ptr<float>(),
        conv_desc,
        bwd_data_algo,
        workspace, workspace_size,
        &beta,
        x_desc, x.data_ptr<float>());
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionBackwardData failed");
    
    // Add bias if provided
    if (bias.defined()) {
        // Bias add: output[n,c,l] += bias[c]
        // We can use cudnnAddTensor
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   1, C_out, 1, 1);
        float beta_add = 1.0f;
        status = cudnnAddTensor(cudnn_handle,
                                &alpha,
                                bias_desc, bias.data_ptr<float>(),
                                &beta_add,
                                y_desc, output.data_ptr<float>());
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnAddTensor failed");
        cudnnDestroyTensorDescriptor(bias_desc);
    }
    
    // Cleanup
    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
    
    return output;
}

void register_conv_transpose1d(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("stride"),
          py::arg("padding"),
          py::arg("output_padding"));
}

REGISTER_BINDING(conv_transpose1d, register_conv_transpose1d);
