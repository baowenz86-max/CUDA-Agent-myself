#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare the softmax launcher from custom_ops.cu
extern "C" void softmax_forward_float32(
    const float* input,
    float* output,
    int B, int C, int L,
    long long input_stride_b,
    long long input_stride_c,
    long long input_stride_l,
    cudaStream_t stream
);

// Helper to check cuDNN status
#define CHECK_CUDNN(err) \
    do { \
        cudnnStatus_t err_ = (err); \
        TORCH_CHECK(err_ == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(err_)); \
    } while (0)

// Helper to create cuDNN tensor descriptor for NCHW or NHWC
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, K)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (C_out)");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");

    int64_t B = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t K = weight.size(2);

    // Compute output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + K + output_padding;
    TORCH_CHECK(L_out > 0, "computed output length is non-positive");

    // Ensure input is contiguous; if not, make contiguous
    auto input_contig = input.is_contiguous() ? input : input.contiguous();
    auto weight_contig = weight.is_contiguous() ? weight : weight.contiguous();
    auto bias_contig = bias.is_contiguous() ? bias : bias.contiguous();

    // Create output tensor
    auto output = torch::empty({B, C_out, L_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // We will use cuDNN's transposed convolution (deconvolution) API.
    // cuDNN expects tensors in NCHW or NHWC. We'll use NCHW (default) but note that
    // for 1D convolution, we can treat it as 2D with height=1. It's often more efficient
    // to use NHWC for better memory locality, but we'll stick with NCHW for simplicity.
    // Since input is [B, C_in, L_in], we map it to NCHW as [B, C_in, 1, L_in].
    // However, cuDNN 1D convolution uses NCHW with height=1. Some versions support
    // CUDNN_TENSOR_NCHW with 3 dimensions directly? Actually, cuDNN requires 4D for
    // convolution descriptors. We'll use 4D with H=1.
    // Let's use CUDNN_TENSOR_NCHW (or NHWC if beneficial) with dims [B, C, 1, L].
    // For transposed conv, cuDNN uses "backward data" operation.

    // Decide layout: NCHW (batch, channels, height=1, width)
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, dx_desc, dy_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dx_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dy_desc));

    // For backward data: dy is input to backward data (i.e., x in transposed conv),
    // dx is output (i.e., y in transposed conv).
    // Set dimensions: batch, channels, height=1, width=L
    int dimA[4] = {static_cast<int>(B), static_cast<int>(C_in), 1, static_cast<int>(L_in)};
    int strideA[4] = {static_cast<int>(C_in * L_in), static_cast<int>(L_in), static_cast<int>(L_in), 1};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, format, CUDNN_DATA_FLOAT,
                                           dimA[0], dimA[1], dimA[2], dimA[3]));

    int dimB[4] = {static_cast<int>(B), static_cast<int>(C_out), 1, static_cast<int>(L_out)};
    int strideB[4] = {static_cast<int>(C_out * L_out), static_cast<int>(L_out), static_cast<int>(L_out), 1};
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, format, CUDNN_DATA_FLOAT,
                                           dimB[0], dimB[1], dimB[2], dimB[3]));

    // Filter descriptor: weight is [C_in, C_out, K] but cuDNN expects [C_out, C_in, K]
    // for convolution filter. For transposed conv, the weight layout in cuDNN is
    // [C_out, C_in, filter_height, filter_width] (or reversed). Actually, for
    // cudnnConvolutionBackwardData, the filter is the same as for forward convolution,
    // i.e., [C_out, C_in, H, W]. We need to transpose weight from [C_in, C_out, K]
    // to [C_out, C_in, K] and possibly add a singleton dimension.
    // We can create a transposed weight tensor as contiguous copy.
    auto weight_t_contig = weight_contig.permute({1, 0, 2}).contiguous(); // [C_out, C_in, K]
    // Now add height dimension: [C_out, C_in, 1, K]
    auto weight_4d = weight_t_contig.unsqueeze(2); // [C_out, C_in, 1, K]
    TORCH_CHECK(weight_4d.is_contiguous(), "weight_4d must be contiguous");

    cudnnFilterDescriptor_t w_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&w_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, format,
                                           static_cast<int>(C_out), static_cast<int>(C_in), 1, static_cast<int>(K)));

    // Convolution descriptor for transposed convolution (backward data)
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    // For 1D, pad height=0, width=padding; stride height=1, width=stride; dilation height=1, width=1
    int padA[2] = {0, static_cast<int>(padding)};
    int strideA_conv[2] = {1, static_cast<int>(stride)};
    int dilationA[2] = {1, 1};
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padA[0], padA[1],
                                                strideA_conv[0], strideA_conv[1],
                                                dilationA[0], dilationA[1],
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Note: For transposed convolution, cuDNN's backward data uses the same convolution
    // descriptor but with the roles of input and output reversed. The padding and stride
    // parameters are interpreted as the forward convolution's parameters. This matches
    // PyTorch's conv_transpose1d semantics (except output_padding, which is handled separately).

    // Find algorithm
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0; // default
    // We could use cudnnFindConvolutionBackwardDataAlgorithm for better performance,
    // but to keep simple, use deterministic algorithm. However, for performance, we should
    // choose a good algorithm. We'll use the workspace-free algorithm if possible.
    size_t workspace_size = 0;
    void* workspace = nullptr;

    // Try to get a recommended algorithm
    int requested_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results[1];
    cudnnFindConvolutionBackwardDataAlgorithm(
        cudnn_handle, w_desc, y_desc, conv_desc, x_desc, 1, &requested_algo_count, perf_results);
    if (requested_algo_count > 0 && perf_results[0].status == CUDNN_STATUS_SUCCESS) {
        bwd_data_algo = perf_results[0].algo;
        workspace_size = perf_results[0].memory;
    } else {
        bwd_data_algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        workspace_size = 0;
    }

    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Perform backward data convolution: computes dx (output) from dy (input) and w
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        w_desc, weight_4d.data_ptr<float>(),
        y_desc, input_contig.data_ptr<float>(),
        conv_desc,
        bwd_data_algo,
        workspace, workspace_size,
        &beta,
        x_desc, output.data_ptr<float>()
    ));

    if (workspace_size > 0) {
        cudaFree(workspace);
    }

    // Now add bias (if provided, we assume bias is always provided as per contract)
    // Bias shape: [C_out]. We need to add to each channel across all batches and positions.
    // We can do this using a simple CUDA kernel, but since we are in binding file,
    // we cannot include CUDA kernels. Instead, we can use cuDNN's tensor transform with addition.
    // However, simpler: we can use cublas or just a custom kernel launcher from .cu.
    // Since we already have a .cu file with softmax, we could add a bias addition kernel there,
    // but the task says we are only modifying .cpp files now. So we need to implement bias add
    // without CUDA kernel. One option is to use a second cuDNN operation: we can use
    // cudnnAddTensor to add bias to the output.
    // cudnnAddTensor adds a tensor with broadcast. We need to broadcast bias from [C_out] to [B,C_out,1,L_out].
    // But cudnnAddTensor requires the bias tensor descriptor to match dimensions and strides.
    // We can create a tensor descriptor for bias with shape [1, C_out, 1, 1] and use cudnnAddTensor
    // with appropriate strides to broadcast.
    cudnnTensorDescriptor_t bias_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
    // Set bias tensor as 4D with dimensions [1, C_out, 1, 1], so it broadcasts over B and L.
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, format, CUDNN_DATA_FLOAT,
                                           1, static_cast<int>(C_out), 1, 1));
    // Note: cudnnAddTensor expects the bias tensor to have the same number of dimensions,
    // and it will broadcast over dimensions with size 1. We'll use alpha=1, beta=1.
    float alpha_add = 1.0f, beta_add = 1.0f;
    CHECK_CUDNN(cudnnAddTensor(cudnn_handle,
                               &alpha_add,
                               bias_desc, bias_contig.data_ptr<float>(),
                               &beta_add,
                               y_desc, output.data_ptr<float>()));

    // Cleanup descriptors
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(dx_desc);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

torch::Tensor softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "dim out of range");

    // For this task, we specifically handle dim=1 for 3D input, but we can support
    // any dim by making contiguous and calling generic kernel? The .cu only has
    // specialized kernel for dim=1 and 3D. We'll restrict to that case.
    TORCH_CHECK(input.dim() == 3, "softmax currently only supports 3D tensors");
    TORCH_CHECK(dim == 1, "softmax currently only supports dim=1");

    int64_t B = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    // We can handle arbitrary strides by passing them to the launcher.
    // The launcher expects strides in elements.
    auto output = torch::empty_like(input);

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Pass strides in elements (not bytes)
    long long stride_b = input.stride(0);
    long long stride_c = input.stride(1);
    long long stride_l = input.stride(2);

    softmax_forward_float32(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(L),
        stride_b,
        stride_c,
        stride_l,
        stream
    );

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d", &conv_transpose1d_forward,
          "1D transposed convolution (cuDNN)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("output_padding") = 0);
    m.def("softmax", &softmax_forward,
          "Softmax along a dimension",
          pybind11::arg("input"),
          pybind11::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
