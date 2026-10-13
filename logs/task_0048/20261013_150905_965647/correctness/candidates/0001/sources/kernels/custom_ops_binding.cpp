// kernels/custom_ops_binding.cpp
// Binding for custom operators.
// Contains PyTorch tensor wrappers and cuDNN implementations for conv operations.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// ======================== Non-convolution launcher declarations ========================

extern "C" void adaptive_avg_pool2d_5d_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int depth,
    int height,
    int width,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_d,
    long long input_stride_h,
    long long input_stride_w,
    cudaStream_t stream
);

extern "C" void leaky_relu_launcher(
    const float* input,
    float* output,
    int total,
    float negative_slope,
    cudaStream_t stream
);

extern "C" void bias_add_launcher(
    const float* input,
    float* output,
    const float* bias,
    int total,
    int channels,
    int depth,
    int height,
    int width,
    cudaStream_t stream
);

extern "C" void clamp_launcher(
    const float* input,
    float* output,
    int total,
    float min_val,
    float max_val,
    cudaStream_t stream
);

// ======================== Helper functions ========================

static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnCreate(&handle);
    }
    return handle;
}

static void check_cudnn(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

static int get_int_from_optional_bias(const torch::Tensor& bias) {
    if (bias.defined() && bias.numel() > 0) return 1;
    return 0;
}

// ======================== Operator wrappers ========================

// 1. Custom Adaptive Average Pooling 2D on 5D tensor
torch::Tensor custom_adaptive_avg_pool2d_5d(torch::Tensor x, int64_t output_size, int64_t output_size_y) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 5, "Input must be 5D");
    TORCH_CHECK(output_size == 1 && output_size_y == 1, "Only output size (1,1) supported");

    int batch = x.size(0);
    int channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);

    // Output shape: [batch, channels, depth, 1, 1]
    auto output = torch::empty({batch, channels, depth, 1, 1}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides (in elements) from input tensor, handling non-contiguous cases.
    long long stride_b = x.stride(0);
    long long stride_c = x.stride(1);
    long long stride_d = x.stride(2);
    long long stride_h = x.stride(3);
    long long stride_w = x.stride(4);

    adaptive_avg_pool2d_5d_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, channels, depth, height, width,
        stride_b, stride_c, stride_d, stride_h, stride_w,
        stream
    );

    return output;
}

// 2. Custom Conv3d using cuDNN
torch::Tensor custom_conv3d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(input.dim() == 5, "Input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    int out_channels = weight.size(0);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int out_depth = (in_depth + 2 * padding - kD) / stride + 1;
    int out_height = (in_height + 2 * padding - kH) / stride + 1;
    int out_width = (in_width + 2 * padding - kW) / stride + 1;

    TORCH_CHECK(out_depth > 0 && out_height > 0 && out_width > 0, "Invalid output dimensions");

    auto output = torch::empty({batch, out_channels, out_depth, out_height, out_width}, input.options());

    cudnnHandle_t handle = get_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(handle, stream);

    // Create tensor descriptors
    cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors (NCDHW format)
    int in_dims[5] = {batch, in_channels, in_depth, in_height, in_width};
    int in_strides[5] = {in_channels * in_depth * in_height * in_width,
                         in_depth * in_height * in_width,
                         in_height * in_width, in_width, 1};
    cudnnSetTensorNdDescriptor(in_desc, CUDNN_DATA_FLOAT, 5, in_dims, in_strides);

    int out_dims[5] = {batch, out_channels, out_depth, out_height, out_width};
    int out_strides[5] = {out_channels * out_depth * out_height * out_width,
                          out_depth * out_height * out_width,
                          out_height * out_width, out_width, 1};
    cudnnSetTensorNdDescriptor(out_desc, CUDNN_DATA_FLOAT, 5, out_dims, out_strides);

    // Bias descriptor (one per channel)
    int bias_dims[5] = {1, out_channels, 1, 1, 1};
    int bias_strides[5] = {out_channels, 1, 1, 1, 1};
    cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 5, bias_dims, bias_strides);

    // Filter descriptor
    int filter_dims[5] = {out_channels, in_channels, kD, kH, kW};
    cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filter_dims);

    // Convolution descriptor
    int pad[3] = {static_cast<int>(padding), static_cast<int>(padding), static_cast<int>(padding)};
    int stride_arr[3] = {static_cast<int>(stride), static_cast<int>(stride), static_cast<int>(stride)};
    int dilation[3] = {1, 1, 1};
    cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad, stride_arr, dilation, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // Find algorithm
    cudnnConvolutionFwdAlgo_t algo;
    const int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results[1];
    check_cudnn(cudnnFindConvolutionForwardAlgorithm(
        handle, in_desc, filter_desc, conv_desc, out_desc,
        requested_algo_count, &returned_algo_count, perf_results),
        "cudnnFindConvolutionForwardAlgorithm failed");
    algo = perf_results[0].algo;

    // Allocate workspace
    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, in_desc, filter_desc, conv_desc, out_desc, algo, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Execute
    float alpha = 1.0f;
    float beta = 0.0f;
    if (bias.defined()) {
        check_cudnn(cudnnConvolutionForward(
            handle,
            &alpha,
            in_desc, input.data_ptr<float>(),
            filter_desc, weight.data_ptr<float>(),
            conv_desc, algo,
            workspace, workspace_size,
            &beta,
            out_desc, output.data_ptr<float>()),
            "cudnnConvolutionForward failed");
        // Add bias
        check_cudnn(cudnnAddTensor(handle, &alpha, bias_desc, bias.data_ptr<float>(), &alpha, out_desc, output.data_ptr<float>()),
                    "cudnnAddTensor bias failed");
    } else {
        check_cudnn(cudnnConvolutionForward(
            handle,
            &alpha,
            in_desc, input.data_ptr<float>(),
            filter_desc, weight.data_ptr<float>(),
            conv_desc, algo,
            workspace, workspace_size,
            &beta,
            out_desc, output.data_ptr<float>()),
            "cudnnConvolutionForward failed");
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return output;
}

// 3. Custom LeakyReLU
torch::Tensor custom_leaky_relu(torch::Tensor x, double negative_slope) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous for elementwise kernel");

    auto output = torch::empty_like(x);
    int total = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    leaky_relu_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        static_cast<float>(negative_slope),
        stream
    );

    return output;
}

// 4. Custom Bias Add
torch::Tensor custom_bias_add(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "Input x must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 5, "x must be 5D");
    TORCH_CHECK(bias.dim() == 4, "bias must be 4D");
    TORCH_CHECK(bias.size(0) == x.size(1), "bias channel dimension must match x channel dimension");

    int batch = x.size(0);
    int channels = x.size(1);
    int depth = x.size(2);
    int height = x.size(3);
    int width = x.size(4);

    auto output = torch::empty_like(x);
    int total = x.numel();

    // Ensure x is contiguous; bias is contiguous by requirement but we can make it contiguous anyway.
    auto x_contig = x.contiguous();
    auto bias_contig = bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bias_add_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        total,
        channels,
        depth,
        height,
        width,
        stream
    );

    return output;
}

// 5. Custom Clamp
torch::Tensor custom_clamp(torch::Tensor x, double min_val, double max_val) {
    TORCH_CHECK(x.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous for elementwise kernel");

    auto output = torch::empty_like(x);
    int total = x.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    clamp_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        total,
        static_cast<float>(min_val),
        static_cast<float>(max_val),
        stream
    );

    return output;
}

// 6. Custom ConvTranspose3d using cuDNN
torch::Tensor custom_conv_transpose3d(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "Bias must be CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    }
    TORCH_CHECK(input.dim() == 5, "Input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "Weight must be 5D");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int in_depth = input.size(2);
    int in_height = input.size(3);
    int in_width = input.size(4);

    // For ConvTranspose, weight shape: [in_channels, out_channels, kD, kH, kW]
    int out_channels = weight.size(1);
    int kD = weight.size(2);
    int kH = weight.size(3);
    int kW = weight.size(4);

    int out_depth = (in_depth - 1) * stride - 2 * padding + kD + output_padding;
    int out_height = (in_height - 1) * stride - 2 * padding + kH + output_padding;
    int out_width = (in_width - 1) * stride - 2 * padding + kW + output_padding;

    TORCH_CHECK(out_depth > 0 && out_height > 0 && out_width > 0, "Invalid output dimensions");

    auto output = torch::empty({batch, out_channels, out_depth, out_height, out_width}, input.options());

    cudnnHandle_t handle = get_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(handle, stream);

    // Create tensor descriptors
    cudnnTensorDescriptor_t in_desc, out_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&in_desc);
    cudnnCreateTensorDescriptor(&out_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    int in_dims[5] = {batch, in_channels, in_depth, in_height, in_width};
    int in_strides[5] = {in_channels * in_depth * in_height * in_width,
                         in_depth * in_height * in_width,
                         in_height * in_width, in_width, 1};
    cudnnSetTensorNdDescriptor(in_desc, CUDNN_DATA_FLOAT, 5, in_dims, in_strides);

    int out_dims[5] = {batch, out_channels, out_depth, out_height, out_width};
    int out_strides[5] = {out_channels * out_depth * out_height * out_width,
                          out_depth * out_height * out_width,
                          out_height * out_width, out_width, 1};
    cudnnSetTensorNdDescriptor(out_desc, CUDNN_DATA_FLOAT, 5, out_dims, out_strides);

    // Bias descriptor
    int bias_dims[5] = {1, out_channels, 1, 1, 1};
    int bias_strides[5] = {out_channels, 1, 1, 1, 1};
    cudnnSetTensorNdDescriptor(bias_desc, CUDNN_DATA_FLOAT, 5, bias_dims, bias_strides);

    // Filter descriptor (note: for cuDNN transposed conv, filter is still [out_channels, in_channels, kD, kH, kW])
    int filter_dims[5] = {out_channels, in_channels, kD, kH, kW};
    cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filter_dims);

    // Convolution descriptor with CUDNN_CROSS_CORRELATION? Actually default is cross-correlation.
    int pad[3] = {static_cast<int>(padding), static_cast<int>(padding), static_cast<int>(padding)};
    int stride_arr[3] = {static_cast<int>(stride), static_cast<int>(stride), static_cast<int>(stride)};
    int dilation[3] = {1, 1, 1};
    cudnnSetConvolutionNdDescriptor(conv_desc, 3, pad, stride_arr, dilation, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    // Use cudnnConvolutionBackwardData to implement transposed convolution.
    // alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // Find algorithm for backward data
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo;
    const int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results[1];
    check_cudnn(cudnnFindConvolutionBackwardDataAlgorithm(
        handle, filter_desc, out_desc, conv_desc, in_desc,
        requested_algo_count, &returned_algo_count, perf_results),
        "cudnnFindConvolutionBackwardDataAlgorithm failed");
    bwd_data_algo = perf_results[0].algo;

    size_t workspace_size = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(handle, filter_desc, out_desc, conv_desc, in_desc, bwd_data_algo, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Compute: output = conv_transpose(input, weight) using backward data.
    // Note: input tensor is considered the "dy" of backward data, output is "dx".
    // cuDNN expects filter to be [out_channels, in_channels, ...] which matches weight for transposed conv.
    check_cudnn(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filter_desc, weight.data_ptr<float>(),
        out_desc, input.data_ptr<float>(),
        conv_desc, bwd_data_algo,
        workspace, workspace_size,
        &beta,
        in_desc, output.data_ptr<float>()),
        "cudnnConvolutionBackwardData failed");

    // Add bias if present
    if (bias.defined()) {
        check_cudnn(cudnnAddTensor(handle, &alpha, bias_desc, bias.data_ptr<float>(), &alpha, in_desc, output.data_ptr<float>()),
                    "cudnnAddTensor bias failed");
    }

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return output;
}

// ======================== Registration ========================

void register_custom_ops(pybind11::module& m) {
    m.def("custom_adaptive_avg_pool2d_5d", &custom_adaptive_avg_pool2d_5d,
          "Adaptive average pooling over last two dims to (1,1)",
          py::arg("x"), py::arg("output_size"), py::arg("output_size_y"));

    m.def("custom_conv3d", &custom_conv3d,
          "3D convolution using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"));

    m.def("custom_leaky_relu", &custom_leaky_relu,
          "LeakyReLU elementwise",
          py::arg("x"), py::arg("negative_slope"));

    m.def("custom_bias_add", &custom_bias_add,
          "Add bias to each channel",
          py::arg("x"), py::arg("bias"));

    m.def("custom_clamp", &custom_clamp,
          "Clamp values between min and max",
          py::arg("x"), py::arg("min_val"), py::arg("max_val"));

    m.def("custom_conv_transpose3d", &custom_conv_transpose3d,
          "3D transposed convolution using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
