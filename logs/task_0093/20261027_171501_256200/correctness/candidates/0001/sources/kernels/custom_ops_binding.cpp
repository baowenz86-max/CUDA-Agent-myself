#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void custom_mul_launcher(const float* x,
                                    const float* bias,
                                    float* output,
                                    int64_t batch_size,
                                    int64_t out_channels,
                                    int64_t length_out,
                                    cudaStream_t stream);

// Helper to check cuDNN status
void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// cuDNN implementation of ConvTranspose1d
// Input: x (N, C, L), weight (C, M, K), bias (M)
// Output: y (N, M, L_out)
// L_out = (L-1)*stride - 2*padding + kernel_size + output_padding
torch::Tensor custom_conv_transpose1d(torch::Tensor x,
                                      torch::Tensor weight,
                                      torch::Tensor bias,
                                      int64_t stride,
                                      int64_t padding,
                                      int64_t output_padding) {
    // Validate inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");

    int N = x.size(0);
    int C = x.size(1);
    int L = x.size(2);
    int K = weight.size(2);
    int M = weight.size(1);
    TORCH_CHECK(weight.size(0) == C, "weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == M, "bias size mismatch");

    // Ensure tensors are contiguous for cuDNN (cuDNN may handle strides but simple is safer)
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    if (!weight.is_contiguous()) {
        weight = weight.contiguous();
    }
    if (!bias.is_contiguous()) {
        bias = bias.contiguous();
    }

    // Compute output length
    int L_out = (L - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "output length must be positive");

    auto output = torch::empty({N, M, L_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    checkCudnn(cudnnCreate(&cudnn_handle), "cudnnCreate failed");
    cudnnSetStream(cudnn_handle, stream);

    // Set up tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    checkCudnn(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x failed");
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y failed");

    // cuDNN layout: NCHW-like, but for 1D we use N,C,L with dimensions as N,C,L,1 or treat L as W?
    // cuDNN supports only 4D and 5D tensors. For 1D convolution, we can use 2D with H=1.
    // Or use 1D via cudnnSetTensor4dDescriptor with N,C,L,1.
    // We'll use 4D descriptor: N, C, L, 1
    checkCudnn(cudnnSetTensor4dDescriptor(x_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          N, C, L, 1),
               "cudnnSetTensor4dDescriptor x failed");
    checkCudnn(cudnnSetTensor4dDescriptor(y_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          N, M, L_out, 1),
               "cudnnSetTensor4dDescriptor y failed");

    // Filter descriptor: for transposed convolution, filter layout is (C, M, K) but cuDNN expects
    // (C, M, kH, kW) for 2D. For 1D, we treat kernel as 1xK, so kH=1, kW=K.
    cudnnFilterDescriptor_t w_desc;
    checkCudnn(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor failed");
    checkCudnn(cudnnSetFilter4dDescriptor(w_desc,
                                          CUDNN_DATA_FLOAT,
                                          CUDNN_TENSOR_NCHW,
                                          C, M, 1, K),
               "cudnnSetFilter4dDescriptor failed");

    // Convolution descriptor for transposed convolution
    cudnnConvolutionDescriptor_t conv_desc;
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");
    checkCudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               padding, 0,  // pad_h=1? Actually pad_h=padding, pad_w=padding? For 1D we set pad_h=padding? Let's set pad_h=0, pad_w=padding
                                               stride, 1,   // stride_h=1? stride_w=stride
                                               1, 1,        // dilation
                                               CUDNN_CROSS_CORRELATION,
                                               CUDNN_DATA_FLOAT),
               "cudnnSetConvolution2dDescriptor failed");

    // For conv_transpose, we need to set output_padding. cuDNN doesn't have direct output_padding in descriptor.
    // Instead, we can use cudnnConvolutionBackwardData which is equivalent to transposed convolution with output_padding.
    // The output_padding can be handled by adjusting the output size manually? Actually cuDNN's 
    // cudnnConvolutionBackwardData computes gradient wrt input, which is transpose convolution, but output size is determined by input size and convolution parameters.
    // However, for stride > 1, the output size may not be uniquely determined; cuDNN has cudnnConvolutionBackwardData with the option to specify desired output dimensions via the output tensor descriptor.
    // We can use cudnnConvolutionBackwardData with the given output descriptor (which includes the larger L_out due to output_padding).
    // The formula for backward data output size is: (output_size_y[0] - 1) * stride_y + kernel_h - 2 * pad_h, which matches (L-1)*stride + K - 2*padding. But we have extra output_padding, so we need to specify the output tensor as the larger one; cuDNN will fill only part? Actually it will compute into the given tensor shape and may leave padding region as zero if not filled.
    // But PyTorch's ConvTranspose1d with output_padding fills the extra elements with zeros, so we can just allocate zero tensor and run the backward data to fill the non-output_padding part.
    // However, there might be a more direct method: use cudnnConvolutionBackwardData with output tensor of the exact desired shape L_out, and the computation will produce values for the first (L-1)*stride + K - 2*padding positions, leaving the last output_padding zeros.
    // We can pre-zero the output to be safe.

    // Zero the output tensor
    output.zero_();

    // Use cudnnConvolutionBackwardData
    float alpha = 1.0f, beta = 0.0f;  // beta=0 since we zeroed output
    checkCudnn(cudnnConvolutionBackwardData(cudnn_handle,
                                            &alpha,
                                            w_desc,
                                            weight.data_ptr<float>(),
                                            x_desc,  // this is actually the input to conv, but for backward data it's the "dy"
                                            x.data_ptr<float>(),
                                            conv_desc,
                                            CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                            nullptr,
                                            0,
                                            &beta,
                                            y_desc,
                                            output.data_ptr<float>()),
               "cudnnConvolutionBackwardData failed");

    // Add bias
    // Bias shape is (M,), need to broadcast to (N, M, L_out)
    // We can use cublas or simple custom kernel? No custom kernel allowed in .cpp, but we can use cudaMemset? Actually we can use cudnnAddTensor or cublasSaxpy-like operation.
    // Simpler: use cudnnAddTensor with bias broadcast.
    // Bias tensor descriptor: (1, M, 1, 1) broadcast to (N, M, L_out, 1).
    cudnnTensorDescriptor_t bias_desc;
    checkCudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
    checkCudnn(cudnnSetTensor4dDescriptor(bias_desc,
                                          CUDNN_TENSOR_NCHW,
                                          CUDNN_DATA_FLOAT,
                                          1, M, 1, 1),
               "cudnnSetTensor4dDescriptor bias failed");
    float alpha_bias = 1.0f, beta_bias = 1.0f;
    checkCudnn(cudnnAddTensor(cudnn_handle,
                              &alpha_bias,
                              bias_desc,
                              bias.data_ptr<float>(),
                              &beta_bias,
                              y_desc,
                              output.data_ptr<float>()),
               "cudnnAddTensor failed");

    // Cleanup
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// custom_mul: element-wise multiply with bias of shape (out_channels, 1)
torch::Tensor custom_mul(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(bias.dim() == 2, "bias must be 2D (out_channels, 1)");
    int64_t N = x.size(0);
    int64_t C = x.size(1);
    int64_t L = x.size(2);
    TORCH_CHECK(bias.size(0) == C, "bias size[0] must equal channels");
    TORCH_CHECK(bias.size(1) == 1, "bias size[1] must be 1");

    auto output = torch::empty_like(x);

    // Ensure contiguous (kernel assumes contiguous)
    auto x_contig = x.is_contiguous() ? x : x.contiguous();
    auto bias_contig = bias.is_contiguous() ? bias : bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    custom_mul_launcher(
        x_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, L,
        stream
    );

    // Check for CUDA errors after launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("custom_mul kernel launch failed: ") + cudaGetErrorString(err));
    }

    return output;
}

// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv_transpose1d", &custom_conv_transpose1d,
          "ConvTranspose1d using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("custom_mul", &custom_mul,
          "Element-wise multiply with bias broadcast",
          py::arg("x"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);