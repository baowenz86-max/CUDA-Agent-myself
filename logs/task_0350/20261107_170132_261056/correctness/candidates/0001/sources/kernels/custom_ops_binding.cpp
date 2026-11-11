// custom_ops_binding.cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void relu_forward_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream);

extern "C" void max_pool1d_forward_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int length_in,
    int length_out,
    int kernel_size,
    int stride,
    int padding,
    cudaStream_t stream);

// cuDNN helper: check status
#define CHECK_CUDNN(expr) \
    do { \
        cudnnStatus_t status = (expr); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
        } \
    } while (0)

// Helper: get current cuDNN handle, reuse via static
static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUDNN(cudnnCreate(&handle));
    }
    return handle;
}

// ConvTranspose1d forward using cuDNN
// Note: cuDNN treats 1D conv as 2D with height=1
at::Tensor conv_transpose1d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Validate inputs
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D: [batch, channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D: [in_channels, out_channels, kernel_size]");

    // Ensure contiguous
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    // Dimensions
    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t in_length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(1);
    int64_t kernel_size = weight_contig.size(2);

    // Compute output length: L_out = (L_in - 1)*stride - 2*padding + kernel_size + output_padding
    int64_t out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "output length must be positive");

    // Output tensor [batch, out_channels, out_length]
    auto output = at::empty({batch, out_channels, out_length}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle
    cudnnHandle_t handle = get_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Create tensor descriptors: use NCHW layout with H=1
    cudnnTensorDescriptor_t input_desc, output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));

    // Descriptor shapes: [N, C, H, W] with H=1, W=length
    // For input: [batch, in_channels, 1, in_length]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)batch,
        (int)in_channels,
        1, // height
        (int)in_length // width
    ));
    // For output: [batch, out_channels, 1, out_length]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)batch,
        (int)out_channels,
        1,
        (int)out_length
    ));

    // Filter descriptor: for deconvolution, filter dims are [in_channels, out_channels, kernel_h, kernel_w]
    // Here kernel_h=1, kernel_w=kernel_size
    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        (int)in_channels,
        (int)out_channels,
        1, // kernel height
        (int)kernel_size // kernel width
    ));

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    // For deconvolution, set padding, stride, and output padding
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        (int)padding, // pad_h
        (int)padding, // pad_w
        (int)stride, // u (vertical stride)
        (int)stride, // v (horizontal stride)
        1, // dilation_h
        1, // dilation_w
        CUDNN_CROSS_CORRELATION, // mode
        CUDNN_DATA_FLOAT
    ));
    // Note: cuDNN deconvolution uses cudnnSetConvolution2dDescriptor but with 
    // cudnnConvolutionBackwardData or cudnnConvolutionBackwardDataPrecomputed?
    // Actually for deconvolution, use cudnnConvolutionBackwardData with stride/padding

    // For true deconvolution, set the descriptor via cudnnSetConvolution2dDescriptor with 
    // sensible values and use cudnnSetConvolutionGroupCount if needed (not).
    // However, PyTorch ConvTranspose1d is implemented as gradient of Conv1d w.r.t. input.
    // cuDNN provides cudnnConvolutionBackwardData which computes the gradient of input given output grad
    // and weight. We can use that to emulate transposed convolution.
    // Here we will call cudnnConvolutionBackwardData with appropriate scaling.
    
    // We need to set the descriptor for the *forward* convolution that would produce the output gradient.
    // This forward convolution has input channels = out_channels, output channels = in_channels,
    // kernel = [1, kernel_size], stride = stride, padding = padding.
    // But cuDNN's backward data expects filter shape [out_channels, in_channels, kernel_h, kernel_w],
    // which matches weight layout [in_channels, out_channels, kernel_size] if we transpose?
    // PyTorch ConvTranspose1d weight has shape [in_channels, out_channels, kernel_size].
    // The corresponding Conv1d (non-transposed) weight would be [out_channels, in_channels, kernel_size].
    // So we need to pass weight as [in_channels, out_channels, kernel_size] but cuDNN expects 
    // [out_channels, in_channels, kernel_size]? Let's check.
    // cudnnConvolutionBackwardData: dy is output grad (shape of output of forward conv),
    // w is filter of forward conv (shape [out_channels, in_channels, kernel_h, kernel_w]),
    // dx is input grad (shape of input of forward conv).
    // For transposed conv, input x (to convtranspose) corresponds to dy (output grad of forward conv),
    // output y corresponds to dx (input grad of forward conv),
    // weight (convtranspose) corresponds to w (filter of forward conv) but with shape [in_channels, out_channels, kernel].
    // So we need to transpose the weight to [out_channels, in_channels, kernel_size] or use a stride trick.
    // Alternatively, we can use cudnnConvolutionBackwardData with filter descriptor set to 
    // [out_channels, in_channels, 1, kernel_size] and pass the transposed weight.
    // For simplicity and performance, we can assume weight is contiguous and create a transposed copy.
    
    // Create transposed weight: [out_channels, in_channels, kernel_size]
    auto weight_transposed = weight_contig.permute({1, 0, 2}).contiguous();
    
    // Filter descriptor for backward data expects [out_channels, in_channels, kernel_h, kernel_w]
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        (int)out_channels,
        (int)in_channels,
        1,
        (int)kernel_size
    ));
    
    // Convolution descriptor for forward conv must have stride and padding same as transposed conv's stride/padding
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        (int)padding,
        (int)padding,
        (int)stride,
        (int)stride,
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));
    
    // Compute algorithm
    cudnnConvolutionBwdDataAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        handle,
        filter_desc,
        output_desc, // dy shape = output_desc (we set output_desc to output of transposed conv, which is dx shape)
        conv_desc,
        input_desc, // dx shape = input_desc (we set input_desc to input of transposed conv, which is dy shape)
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo
    ));
    
    // Workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        filter_desc,
        output_desc,
        conv_desc,
        input_desc,
        algo,
        &workspace_size
    ));
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }
    
    // Perform backward data: output = dx, input = dy, weight_transposed
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filter_desc,
        weight_transposed.data_ptr<float>(),
        output_desc,  // dy descriptor (x)
        input_contig.data_ptr<float>(), // dy data (x)
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        input_desc,  // dx descriptor (y)
        output.data_ptr<float>() // dx data (y)
    ));
    
    // Add bias if provided
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == at::kFloat, "bias must be CUDA float32");
        TORCH_CHECK(bias.numel() == out_channels, "bias size must match out_channels");
        // Add bias to each channel across length and batch
        // For each (n,c,l): output[n,c,l] += bias[c]
        // Use a simple CUDA kernel or use cuBLAS? Since we are in .cpp, we can't launch kernels.
        // But we can use cudaMemcpy2D? Not straightforward. Better: use a custom CUDA kernel in .cu.
        // However, requirement: .cpp only cuDNN/cuBLAS, no custom kernels. We can use cuBLAS to add bias?
        // Bias addition is just adding a constant vector to each channel. We can use cudnnAddTensor.
        // We'll create a bias descriptor of shape [1, out_channels, 1, 1] and use cudnnAddTensor.
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            bias_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1, out_channels, 1, 1
        ));
        float alpha_bias = 1.0f;
        float beta_bias = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(
            handle,
            &alpha_bias,
            bias_desc,
            bias.contiguous().data_ptr<float>(),
            &beta_bias,
            output_desc,
            output.data_ptr<float>()
        ));
    }
    
    // Cleanup
    if (workspace) cudaFree(workspace);
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    
    return output;
}

// Conv1d forward using cuDNN
at::Tensor conv1d_forward(
    at::Tensor input,
    at::Tensor weight,
    at::Tensor bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(weight.dtype() == at::kFloat, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D: [batch, channels, length]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D: [in_channels, out_channels, kernel_size]");

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    int64_t batch = input_contig.size(0);
    int64_t in_channels = input_contig.size(1);
    int64_t in_length = input_contig.size(2);
    int64_t out_channels = weight_contig.size(0);
    int64_t kernel_size = weight_contig.size(2);

    int64_t out_length = (in_length + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(out_length > 0, "output length must be positive");

    auto output = at::empty({batch, out_channels, out_length}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnHandle_t handle = get_cudnn_handle();
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    cudnnTensorDescriptor_t input_desc, output_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        (int)batch, (int)in_channels, 1, (int)in_length));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        (int)batch, (int)out_channels, 1, (int)out_length));

    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        (int)out_channels, (int)in_channels, 1, (int)kernel_size));

    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
        (int)padding, (int)padding, (int)stride, (int)stride, 1, 1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(handle,
        input_desc, filter_desc, conv_desc, output_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));

    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
        input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size));

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(handle,
        &alpha, input_desc, input_contig.data_ptr<float>(),
        filter_desc, weight_contig.data_ptr<float>(),
        conv_desc, algo, workspace, workspace_size,
        &beta, output_desc, output.data_ptr<float>()));

    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == at::kFloat, "bias must be CUDA float32");
        TORCH_CHECK(bias.numel() == out_channels, "bias size must match out_channels");
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
            1, out_channels, 1, 1));
        float alpha_bias = 1.0f, beta_bias = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(handle, &alpha_bias, bias_desc, bias.contiguous().data_ptr<float>(),
            &beta_bias, output_desc, output.data_ptr<float>()));
    }

    if (workspace) cudaFree(workspace);
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));

    return output;
}

at::Tensor relu_forward(at::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    auto input_contig = input.contiguous();
    auto output = at::empty_like(input_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    long long numel = input_contig.numel();
    relu_forward_launcher(input_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    return output;
}

at::Tensor max_pool1d_forward(
    at::Tensor input,
    int64_t kernel_size,
    int64_t stride,
    int64_t padding) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == at::kFloat, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D: [batch, channels, length]");
    auto input_contig = input.contiguous();
    int64_t batch = input_contig.size(0);
    int64_t channels = input_contig.size(1);
    int64_t length_in = input_contig.size(2);
    int64_t length_out = (length_in + 2 * padding - kernel_size) / stride + 1;
    TORCH_CHECK(length_out > 0, "output length must be positive");
    auto output = at::empty({batch, channels, length_out}, input_contig.options());
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    max_pool1d_forward_launcher(
        input_contig.data_ptr<float>(), output.data_ptr<float>(),
        (int)batch, (int)channels, (int)length_in, (int)length_out,
        (int)kernel_size, (int)stride, (int)padding, stream);
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride") = 2, pybind11::arg("padding") = 1, pybind11::arg("output_padding") = 1);
    m.def("conv1d_forward", &conv1d_forward,
          "Conv1d forward using cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride") = 1, pybind11::arg("padding") = 1);
    m.def("relu_forward", &relu_forward, "ReLU forward");
    m.def("max_pool1d_forward", &max_pool1d_forward, "MaxPool1d forward",
          pybind11::arg("input"), pybind11::arg("kernel_size") = 2,
          pybind11::arg("stride") = 2, pybind11::arg("padding") = 0);
}

REGISTER_BINDING(custom_ops, register_custom_ops);
