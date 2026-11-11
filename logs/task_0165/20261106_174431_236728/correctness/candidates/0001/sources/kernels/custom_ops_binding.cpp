#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>

#include <vector>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void log_softmax_forward_launcher(
    float* output,
    const float* input,
    const int64_t* dims,
    int ndims,
    int dim,
    cudaStream_t stream);

namespace {

cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnCreate(&handle);
    }
    return handle;
}

void check_cuda(cudaError_t status, const char* msg) {
    TORCH_CHECK(status == cudaSuccess, msg, cudaGetErrorString(status));
}

void check_cudnn(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, cudnnGetErrorString(status));
}

// Helper to compute output size for ConvTranspose
template <typename T>
T conv_transpose_out_size(T in_size, T kernel_size, T stride, T padding, T output_padding) {
    return (in_size - 1) * stride - 2 * padding + kernel_size + output_padding;
}

// ConvTranspose1d using cuDNN
// Parameters:
// - input: contiguous float32 CUDA tensor of shape [N, C_in, L_in]
// - weight: contiguous float32 CUDA tensor of shape [C_in, C_out, K]
// - bias: optional contiguous float32 CUDA tensor of shape [C_out], may be None
// - stride: int
// - padding: int
// Returns output tensor of shape [N, C_out, L_out]
torch::Tensor conv_transpose1d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding) {

    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose1d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose1d_forward: weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "conv_transpose1d_forward: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_forward: weight must be contiguous");
    TORCH_CHECK(input.dim() == 3, "conv_transpose1d_forward: input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_forward: weight must be 3D");
    TORCH_CHECK(weight.size(0) == input.size(1), "conv_transpose1d_forward: weight.size(0) must match input.size(1)");
    if (bias.has_value() && bias.value().defined()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "conv_transpose1d_forward: bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "conv_transpose1d_forward: bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "conv_transpose1d_forward: bias must be contiguous");
        TORCH_CHECK(b.dim() == 1, "conv_transpose1d_forward: bias must be 1D");
        TORCH_CHECK(b.size(0) == weight.size(1), "conv_transpose1d_forward: bias.size(0) must match weight.size(1)");
    }

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);
    int64_t output_padding = 0; // No output_padding for ConvTranspose1d in this model (it's not a parameter)
    int64_t L_out = conv_transpose_out_size(L_in, K, stride, padding, output_padding);

    // Prepare output tensor
    auto output = torch::empty({N, C_out, L_out}, input.options());

    cudnnHandle_t handle = get_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream failed");

    // Set up tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor input failed");
    check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor output failed");

    // For 1D convolution, cuDNN uses NCHW format with H=1? Actually ConvTranspose1d can be treated as 2D with height=1.
    // We set tensor format as N,C,1,W and kernel as C_in,C_out,1,K with stride (1,stride), padding (0,padding), output_padding (0,0).
    // But cuDNN does not support 1D convolutions natively, so we can work around by using 2D with trivial height dimension.
    // Alternatively, use cuDNN's 2D transpose with input and output height = 1.
    // We'll follow this approach: set input as [N, C_in, 1, L_in], output as [N, C_out, 1, L_out], filter as [C_in, C_out, 1, K].

    // Set tensor descriptors with NCHW format
    int dims_in[4] = {(int)N, (int)C_in, 1, (int)L_in};
    int strides_in[4] = {(int)(C_in * L_in), (int)L_in, (int)L_in, 1};
    int dims_out[4] = {(int)N, (int)C_out, 1, (int)L_out};
    int strides_out[4] = {(int)(C_out * L_out), (int)L_out, (int)L_out, 1};

    check_cudnn(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          dims_in[0], dims_in[1], dims_in[2], dims_in[3]),
                "cudnnSetTensor4dDescriptor input failed");
    check_cudnn(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          dims_out[0], dims_out[1], dims_out[2], dims_out[3]),
                "cudnnSetTensor4dDescriptor output failed");

    // Filter descriptor
    cudnnFilterDescriptor_t filter_desc;
    check_cudnn(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor failed");
    int filter_dims[4] = {(int)C_in, (int)C_out, 1, (int)K};
    check_cudnn(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]),
                "cudnnSetFilter4dDescriptor failed");

    // Convolution descriptor for transpose
    cudnnConvolutionDescriptor_t conv_desc;
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                               0,           // pad_h
                                               (int)padding, // pad_w
                                               1,           // stride_h
                                               (int)stride, // stride_w
                                               1,           // dilation_h
                                               1,           // dilation_w
                                               CUDNN_CROSS_CORRELATION, // mode (like PyTorch convolution)
                                               CUDNN_DATA_FLOAT),
                "cudnnSetConvolution2dDescriptor failed");

    // cuDNN backward data is equivalent to transpose convolution.
    // alpha/beta for conv
    float alpha = 1.0f, beta = 0.0f;

    // Workspace size query and allocation
    size_t workspace_size = 0;
    check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                             filter_desc,
                                                             output_desc,  // note: dy (gradient w.r.t output) corresponds to input of transpose conv
                                                             conv_desc,
                                                             input_desc,   // dx (gradient w.r.t input) corresponds to output of transpose conv
                                                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                                             &workspace_size),
                "cudnnGetConvolutionBackwardDataWorkspaceSize failed");

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        check_cuda(err, "cudaMalloc workspace failed");
    }

    // Execute transpose convolution: out = weight^T * input + bias (if any)
    // We first compute conv_transpose without bias, then add bias.
    check_cudnn(cudnnConvolutionBackwardData(handle,
                                             &alpha,
                                             filter_desc,
                                             weight.data_ptr<float>(),
                                             output_desc, // dy (input of transpose conv)
                                             input.data_ptr<float>(),
                                             conv_desc,
                                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                                             workspace,
                                             workspace_size,
                                             &beta,
                                             input_desc, // dx (output of transpose conv)
                                             output.data_ptr<float>()),
                "cudnnConvolutionBackwardData failed");

    // Add bias if provided
    if (bias.has_value() && bias.value().defined()) {
        // Add bias to each channel: output[n, c, l] += bias[c]
        // We can use cudnnAddTensor with bias descriptor of shape [1, C_out, 1, 1]
        cudnnTensorDescriptor_t bias_desc;
        check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
        int bias_dims[4] = {1, (int)C_out, 1, 1};
        int bias_strides[4] = {(int)C_out, 1, 1, 1};
        check_cudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              bias_dims[0], bias_dims[1], bias_dims[2], bias_dims[3]),
                    "cudnnSetTensor4dDescriptor bias failed");
        // bias_desc stride settings may not be needed but ensure proper layout
        // We can instead do a simple custom kernel? But we are not allowed custom kernels in .cpp.
        // Use cudnnAddTensor.
        float bias_alpha = 1.0f;
        float bias_beta = 1.0f;
        check_cudnn(cudnnAddTensor(handle,
                                   &bias_alpha,
                                   bias_desc,
                                   bias.value().data_ptr<float>(),
                                   &bias_beta,
                                   output_desc,
                                   output.data_ptr<float>()),
                    "cudnnAddTensor bias failed");
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace_size > 0) {
        cudaFree(workspace);
    }

    return output;
}

// ConvTranspose2d using cuDNN
// Parameters:
// - input: contiguous float32 CUDA tensor of shape [N, C_in, H_in, W_in]
// - weight: contiguous float32 CUDA tensor of shape [C_in, C_out, K_h, K_w]
// - bias: optional contiguous float32 CUDA tensor of shape [C_out], may be None
// - stride: int
// - padding: int
// - output_padding: int
torch::Tensor conv_transpose2d_forward(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    TORCH_CHECK(input.is_cuda(), "conv_transpose2d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose2d_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose2d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose2d_forward: weight must be float32");
    TORCH_CHECK(input.is_contiguous(), "conv_transpose2d_forward: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose2d_forward: weight must be contiguous");
    TORCH_CHECK(input.dim() == 4, "conv_transpose2d_forward: input must be 4D");
    TORCH_CHECK(weight.dim() == 4, "conv_transpose2d_forward: weight must be 4D");
    TORCH_CHECK(weight.size(0) == input.size(1), "conv_transpose2d_forward: weight.size(0) must match input.size(1)");
    if (bias.has_value() && bias.value().defined()) {
        auto b = bias.value();
        TORCH_CHECK(b.is_cuda(), "conv_transpose2d_forward: bias must be CUDA");
        TORCH_CHECK(b.dtype() == torch::kFloat32, "conv_transpose2d_forward: bias must be float32");
        TORCH_CHECK(b.is_contiguous(), "conv_transpose2d_forward: bias must be contiguous");
        TORCH_CHECK(b.dim() == 1, "conv_transpose2d_forward: bias must be 1D");
        TORCH_CHECK(b.size(0) == weight.size(1), "conv_transpose2d_forward: bias.size(0) must match weight.size(1)");
    }

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t H_in = input.size(2);
    int64_t W_in = input.size(3);
    int64_t C_out = weight.size(1);
    int64_t K_h = weight.size(2);
    int64_t K_w = weight.size(3);
    int64_t H_out = conv_transpose_out_size(H_in, K_h, stride, padding, output_padding);
    int64_t W_out = conv_transpose_out_size(W_in, K_w, stride, padding, output_padding);

    auto output = torch::empty({N, C_out, H_out, W_out}, input.options());

    cudnnHandle_t handle = get_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream failed");

    cudnnTensorDescriptor_t input_desc, output_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor input failed");
    check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor output failed");

    // Set tensor descriptors NCHW
    check_cudnn(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          (int)N, (int)C_in, (int)H_in, (int)W_in),
                "cudnnSetTensor4dDescriptor input failed");
    check_cudnn(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                          (int)N, (int)C_out, (int)H_out, (int)W_out),
                "cudnnSetTensor4dDescriptor output failed");

    cudnnFilterDescriptor_t filter_desc;
    check_cudnn(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor failed");
    check_cudnn(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                          (int)C_in, (int)C_out, (int)K_h, (int)K_w),
                "cudnnSetFilter4dDescriptor failed");

    cudnnConvolutionDescriptor_t conv_desc;
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                (int)padding,   // pad_h
                                                (int)padding,   // pad_w
                                                (int)stride,    // stride_h
                                                (int)stride,    // stride_w
                                                1,              // dilation_h
                                                1,              // dilation_w
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT),
                "cudnnSetConvolution2dDescriptor failed");

    float alpha = 1.0f, beta = 0.0f;

    size_t workspace_size = 0;
    // Choose algorithm 0 or 1, we can use get with algo 0 for determinism
    check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(handle,
                                                             filter_desc,
                                                             output_desc,
                                                             conv_desc,
                                                             input_desc,
                                                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                                             &workspace_size),
                "cudnnGetConvolutionBackwardDataWorkspaceSize failed");

    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        check_cuda(err, "cudaMalloc workspace failed");
    }

    check_cudnn(cudnnConvolutionBackwardData(handle,
                                             &alpha,
                                             filter_desc,
                                             weight.data_ptr<float>(),
                                             output_desc,
                                             input.data_ptr<float>(),
                                             conv_desc,
                                             CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                                             workspace,
                                             workspace_size,
                                             &beta,
                                             input_desc,
                                             output.data_ptr<float>()),
                "cudnnConvolutionBackwardData failed");

    // Add bias if provided
    if (bias.has_value() && bias.value().defined()) {
        cudnnTensorDescriptor_t bias_desc;
        check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
        // Bias shape [1, C_out, 1, 1]
        check_cudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                              1, (int)C_out, 1, 1),
                    "cudnnSetTensor4dDescriptor bias failed");
        float bias_alpha = 1.0f;
        float bias_beta = 1.0f;
        check_cudnn(cudnnAddTensor(handle,
                                   &bias_alpha,
                                   bias_desc,
                                   bias.value().data_ptr<float>(),
                                   &bias_beta,
                                   output_desc,
                                   output.data_ptr<float>()),
                    "cudnnAddTensor bias failed");
        cudnnDestroyTensorDescriptor(bias_desc);
    }

    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace_size > 0) {
        cudaFree(workspace);
    }

    return output;
}

// log_softmax binding
torch::Tensor log_softmax_forward(const torch::Tensor& input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "log_softmax_forward: input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "log_softmax_forward: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "log_softmax_forward: input must be contiguous");
    TORCH_CHECK(input.numel() > 0, "log_softmax_forward: input must not be empty");

    auto output = torch::empty_like(input);

    // Prepare dims array for launcher
    int ndims = input.dim();
    std::vector<int64_t> dims_vec(ndims);
    for (int i = 0; i < ndims; ++i) {
        dims_vec[i] = input.size(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    log_softmax_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        dims_vec.data(),
        ndims,
        (int)dim,
        stream);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "log_softmax_forward launcher failed: ", cudaGetErrorString(err));

    return output;
}

} // namespace

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"));
    m.def("conv_transpose2d_forward", &conv_transpose2d_forward,
          "ConvTranspose2d forward using cuDNN",
          pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
    m.def("log_softmax_forward", &log_softmax_forward,
          "LogSoftmax forward",
          pybind11::arg("input"), pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
