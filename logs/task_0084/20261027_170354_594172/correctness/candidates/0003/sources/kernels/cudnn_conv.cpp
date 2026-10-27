#include <torch/types.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Exceptions.h>

#define CHECK_CUDNN(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
    }

torch::Tensor cudnn_conv2d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    int64_t groups) {

    // Input checks
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(!bias.defined() || bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(!bias.defined() || bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D");

    // Get dimensions
    int64_t N = input.size(0);       // batch size
    int64_t C = input.size(1);       // input channels
    int64_t H = input.size(2);       // input height
    int64_t W = input.size(3);       // input width
    int64_t K = weight.size(0);      // output channels
    int64_t R = weight.size(2);      // kernel height
    int64_t S = weight.size(3);      // kernel width

    TORCH_CHECK(weight.size(1) == C / groups, "weight input channels must match input channels / groups");
    TORCH_CHECK(groups > 0 && C % groups == 0 && K % groups == 0, "invalid groups");

    // Compute output dimensions
    int64_t H_out = (H + 2 * padding - dilation * (R - 1) - 1) / stride + 1;
    int64_t W_out = (W + 2 * padding - dilation * (S - 1) - 1) / stride + 1;

    // Allocate output tensor with same dtype/device, but ensure contiguous
    auto output = torch::empty({N, K, H_out, W_out}, input.options());

    // Make tensors contiguous if needed (cuDNN requires contiguous or specific strides, but we can handle non-contiguous via descriptors)
    // For simplicity, use contiguous versions. This does not affect correctness.
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto output_contig = output.contiguous();

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Set up tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set input/output tensor descriptors (NCHW format)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        input_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)N,
        (int)C,
        (int)H,
        (int)W
    ));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        output_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        (int)N,
        (int)K,
        (int)H_out,
        (int)W_out
    ));

    // Set filter descriptor: K, C/groups, R, S
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        filter_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        (int)K,
        (int)(C / groups),
        (int)R,
        (int)S
    ));

    // Set convolution descriptor
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        (int)padding,
        (int)padding,
        (int)stride,
        (int)stride,
        (int)dilation,
        (int)dilation,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, (int)groups));

    // Find best convolution algorithm
    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo
    ));

    // Determine workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size
    ));

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for cuDNN workspace");
        }
    }

    // Perform convolution
    float alpha = 1.0f;
    float beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha,
        input_desc,
        input_contig.data_ptr<float>(),
        filter_desc,
        weight_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        output_desc,
        output_contig.data_ptr<float>()
    ));

    // Add bias if defined
    if (bias.defined()) {
        TORCH_CHECK(bias.numel() == K, "bias size must match output channels");
        // Bias descriptor: N=1, C=K, H=1, W=1
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(
            bias_desc,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_FLOAT,
            1,
            (int)K,
            1,
            1
        ));

        alpha = 1.0f;
        beta = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(
            handle,
            &alpha,
            bias_desc,
            bias.contiguous().data_ptr<float>(),
            &beta,
            output_desc,
            output_contig.data_ptr<float>()
        ));
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);
    if (workspace) cudaFree(workspace);

    // Return output (contiguous tensor)
    return output_contig;
}
