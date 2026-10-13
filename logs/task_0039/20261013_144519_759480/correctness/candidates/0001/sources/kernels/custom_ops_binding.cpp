// Binding file for custom ops: conv2d_bias_bn, avg_pool2d, sigmoid
// Uses cuDNN for convolutions, calls launchers from custom_ops.cu for pooling and sigmoid.
// No torch operators used for computation; only tensor creation and memory management.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void avg_pool2d_launcher(
    const float* input,
    int N, int C, int H, int W,
    int input_n_stride, int input_c_stride, int input_h_stride, int input_w_stride,
    int kernel_size, int stride, int padding,
    float* output, int outH, int outW,
    cudaStream_t stream);

extern "C" void sigmoid_launcher(
    const float* input,
    float* output,
    int numel,
    cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
    } while (0)

// Helper to check cuDNN errors
#define CHECK_CUDNN_ERROR(call) \
    do { \
        cudnnStatus_t status = call; \
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(status)); \
    } while (0)

// Performs Conv2d with cuDNN, adds bias, then BatchNorm eval
// Parameters: input, conv_weight, conv_bias, add_bias, bn_weight, bn_bias,
//             running_mean, running_var, eps, kernel_size, stride, padding
// kernel_size, stride, padding are for avg_pool but unused here; conv uses fixed 3x3 stride 1 padding 1
torch::Tensor conv2d_bias_bn(
    const torch::Tensor& input,
    const torch::Tensor& conv_weight,
    const torch::Tensor& conv_bias,
    const torch::Tensor& add_bias,
    const torch::Tensor& bn_weight,
    const torch::Tensor& bn_bias,
    const torch::Tensor& running_mean,
    const torch::Tensor& running_var,
    double eps,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(conv_weight.is_cuda(), "conv_weight must be a CUDA tensor");
    TORCH_CHECK(conv_bias.is_cuda(), "conv_bias must be a CUDA tensor");
    TORCH_CHECK(add_bias.is_cuda(), "add_bias must be a CUDA tensor");
    TORCH_CHECK(bn_weight.is_cuda(), "bn_weight must be a CUDA tensor");
    TORCH_CHECK(bn_bias.is_cuda(), "bn_bias must be a CUDA tensor");
    TORCH_CHECK(running_mean.is_cuda(), "running_mean must be a CUDA tensor");
    TORCH_CHECK(running_var.is_cuda(), "running_var must be a CUDA tensor");

    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(conv_weight.dtype() == torch::kFloat32, "conv_weight must be float32");
    TORCH_CHECK(conv_bias.dtype() == torch::kFloat32, "conv_bias must be float32");
    TORCH_CHECK(add_bias.dtype() == torch::kFloat32, "add_bias must be float32");
    TORCH_CHECK(bn_weight.dtype() == torch::kFloat32, "bn_weight must be float32");
    TORCH_CHECK(bn_bias.dtype() == torch::kFloat32, "bn_bias must be float32");
    TORCH_CHECK(running_mean.dtype() == torch::kFloat32, "running_mean must be float32");
    TORCH_CHECK(running_var.dtype() == torch::kFloat32, "running_var must be float32");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(conv_weight.dim() == 4, "conv_weight must be 4D");
    TORCH_CHECK(conv_bias.dim() == 1, "conv_bias must be 1D");
    TORCH_CHECK(add_bias.dim() == 3, "add_bias must be 3D (C,1,1)");
    TORCH_CHECK(bn_weight.dim() == 1, "bn_weight must be 1D");
    TORCH_CHECK(bn_bias.dim() == 1, "bn_bias must be 1D");
    TORCH_CHECK(running_mean.dim() == 1, "running_mean must be 1D");
    TORCH_CHECK(running_var.dim() == 1, "running_var must be 1D");

    TORCH_CHECK(input.size(1) == conv_weight.size(1), "input channels must match conv_weight channels");
    TORCH_CHECK(conv_weight.size(0) == add_bias.size(0), "conv out channels must match add_bias size");
    TORCH_CHECK(conv_weight.size(0) == bn_weight.size(0), "conv out channels must match bn size");

    // Ensure contiguous for cuDNN (except input may have strides, we'll handle via cuDNN or make contiguous)
    auto input_contig = input.contiguous();
    auto weight_contig = conv_weight.contiguous();
    auto conv_bias_contig = conv_bias.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t H = input_contig.size(2);
    int64_t W = input_contig.size(3);
    int64_t C_out = weight_contig.size(0);
    int64_t K = weight_contig.size(2); // kernel size (assume square)
    int64_t stride_conv = 1;
    int64_t padding_conv = 1;
    int64_t dilation = 1;

    // Output spatial dims
    int64_t H_out = (H + 2 * padding_conv - dilation * (K - 1) - 1) / stride_conv + 1;
    int64_t W_out = (W + 2 * padding_conv - dilation * (K - 1) - 1) / stride_conv + 1;
    TORCH_CHECK(H_out == H, "Expected conv output H to match input H for this config");
    TORCH_CHECK(W_out == W, "Expected conv output W to match input W for this config");

    // Allocate conv output tensor (will hold conv result + bias add + BN later)
    auto output = torch::empty({N, C_out, H_out, W_out}, input_contig.options());

    // cuDNN handle and descriptors
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN_ERROR(cudnnCreate(&cudnn_handle));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDNN_ERROR(cudnnSetStream(cudnn_handle, stream));

    cudnnTensorDescriptor_t input_desc;
    cudnnTensorDescriptor_t output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors (NCHW format)
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, H, W));
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, H_out, W_out));

    // Filter descriptor: weights are in OIHW order (PyTorch)
    CHECK_CUDNN_ERROR(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, K, K));

    // Convolution descriptor
    CHECK_CUDNN_ERROR(cudnnSetConvolution2dDescriptor(conv_desc, padding_conv, padding_conv, stride_conv, stride_conv, dilation, dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Find fastest algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardAlgorithm(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

    // Workspace
    size_t workspace_size = 0;
    CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA_ERROR(cudaMalloc(&workspace, workspace_size));
    }

    // Perform convolution with alpha=1, beta=0 (no bias yet)
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN_ERROR(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc, input_contig.data_ptr<float>(),
        filter_desc, weight_contig.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        output_desc, output.data_ptr<float>()));

    // Add conv bias if present
    if (conv_bias_contig.numel() > 0) {
        // Use cudnnAddTensor or custom kernel; here use cudnnAddTensor with bias descriptor
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&bias_desc));
        CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
        float alpha_bias = 1.0f, beta_bias = 1.0f;
        CHECK_CUDNN_ERROR(cudnnAddTensor(cudnn_handle, &alpha_bias, bias_desc, conv_bias_contig.data_ptr<float>(), &beta_bias, output_desc, output.data_ptr<float>()));
        CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Add add_bias (shape [C_out,1,1]) broadcast over spatial dims
    // Use cudnnAddTensor similarly
    cudnnTensorDescriptor_t add_bias_desc;
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&add_bias_desc));
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(add_bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
    float alpha_add = 1.0f, beta_add = 1.0f;
    CHECK_CUDNN_ERROR(cudnnAddTensor(cudnn_handle, &alpha_add, add_bias_desc, add_bias.data_ptr<float>(), &beta_add, output_desc, output.data_ptr<float>()));
    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(add_bias_desc));

    // BatchNorm eval mode: y = (x - running_mean) / sqrt(running_var + eps) * bn_weight + bn_bias
    // We'll implement with a custom kernel or use cudnnBatchNormalizationForwardInference.
    // Since we cannot add new .cu kernels per constraints, we can use cudnnBatchNormalizationForwardInference.
    // However, cudnn requires a specific input descriptor and per-activation mode.
    // We'll use cudnnBatchNormalizationForwardInference with mode CUDNN_BATCHNORM_SPATIAL (per-channel).
    // Need to create bn scale/bias tensors from bn_weight and bn_bias, and have BN descriptors.
    cudnnTensorDescriptor_t bn_input_desc = output_desc; // reuse output descriptor (N,C,H,W)
    cudnnTensorDescriptor_t bn_output_desc = output_desc;
    cudnnTensorDescriptor_t bn_scale_bias_mean_var_desc;
    CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(&bn_scale_bias_mean_var_desc));
    CHECK_CUDNN_ERROR(cudnnSetTensor4dDescriptor(bn_scale_bias_mean_var_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));

    // We need factor = bn_weight / sqrt(running_var + eps), and bias = bn_bias - running_mean * factor
    // But cudnnBatchNormalizationForwardInference can take scale and bias directly and does the normalization internally with running estimates.
    // cudnn signature: cudnnBatchNormalizationForwardInference(handle, mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
    // bnScale points to bn_weight, bnBias points to bn_bias, estimatedMean = running_mean, estimatedVariance = running_var.
    // mode = CUDNN_BATCHNORM_SPATIAL (per-activation for NCHW, per-channel normalization).
    // Set alpha=1, beta=0 to overwrite.
    float alpha_bn = 1.0f, beta_bn = 0.0f;
    CHECK_CUDNN_ERROR(cudnnBatchNormalizationForwardInference(
        cudnn_handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha_bn, &beta_bn,
        bn_input_desc, output.data_ptr<float>(),
        bn_output_desc, output.data_ptr<float>(), // in-place
        bn_scale_bias_mean_var_desc,
        bn_weight.data_ptr<float>(),
        bn_bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        eps));

    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(bn_scale_bias_mean_var_desc));

    // Cleanup cuDNN resources
    if (workspace) CHECK_CUDA_ERROR(cudaFree(workspace));
    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN_ERROR(cudnnDestroy(cudnn_handle));

    return output;
}

// AvgPool2d binding
torch::Tensor avg_pool2d(
    const torch::Tensor& input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");

    // input may have arbitrary strides, pass them to launcher
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    int64_t input_n_stride = input.stride(0);
    int64_t input_c_stride = input.stride(1);
    int64_t input_h_stride = input.stride(2);
    int64_t input_w_stride = input.stride(3);

    // Output shape: same as input for kernel=3, stride=1, padding=1, but compute generally
    int64_t outH = (H + 2 * padding - kernel_size) / stride + 1;
    int64_t outW = (W + 2 * padding - kernel_size) / stride + 1;

    auto output = torch::empty({N, C, outH, outW}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    avg_pool2d_launcher(
        input.data_ptr<float>(),
        static_cast<int>(N), static_cast<int>(C), static_cast<int>(H), static_cast<int>(W),
        static_cast<int>(input_n_stride), static_cast<int>(input_c_stride), static_cast<int>(input_h_stride), static_cast<int>(input_w_stride),
        static_cast<int>(kernel_size), static_cast<int>(stride), static_cast<int>(padding),
        output.data_ptr<float>(), static_cast<int>(outH), static_cast<int>(outW),
        stream);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return output;
}

// Sigmoid binding
torch::Tensor sigmoid(const torch::Tensor& input)
{
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    sigmoid_launcher(input.data_ptr<float>(), output.data_ptr<float>(), static_cast<int>(numel), stream);

    CHECK_CUDA_ERROR(cudaGetLastError());
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv2d_bias_bn", &conv2d_bias_bn,
          "Conv2d + bias + BatchNorm (eval) fused",
          pybind11::arg("input"), pybind11::arg("conv_weight"), pybind11::arg("conv_bias"),
          pybind11::arg("add_bias"), pybind11::arg("bn_weight"), pybind11::arg("bn_bias"),
          pybind11::arg("running_mean"), pybind11::arg("running_var"), pybind11::arg("eps"),
          pybind11::arg("kernel_size"), pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("avg_pool2d", &avg_pool2d, "Average pooling 2d",
          pybind11::arg("input"), pybind11::arg("kernel_size"), pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("sigmoid", &sigmoid, "Element-wise sigmoid", pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
