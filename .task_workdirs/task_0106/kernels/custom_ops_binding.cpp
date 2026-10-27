#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void hardswish_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);
extern "C" void softplus_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA_ERR() { \
    cudaError_t err = cudaGetLastError(); \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
}
#define CHECK_CUDNN_ERR(status) { \
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(status)); \
}

// Hardswish forward wrapper
torch::Tensor hardswish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    hardswish_forward_launcher(x_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    CHECK_CUDA_ERR();
    return output;
}

// Softplus forward wrapper
torch::Tensor softplus_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int64_t numel = x_contig.numel();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_forward_launcher(x_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    CHECK_CUDA_ERR();
    return output;
}

// ConvTranspose1d forward using cuDNN
torch::Tensor conv_transpose1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                      int64_t stride, int64_t padding, int64_t output_padding) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "Weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "Bias must be float32");
    TORCH_CHECK(x.dim() == 3, "Input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "Weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(stride > 0 && padding >= 0 && output_padding >= 0, "Invalid stride/padding/output_padding");

    // Make contiguous
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int batch = x_contig.size(0);
    int in_channels = x_contig.size(1);
    int in_length = x_contig.size(2);
    int out_channels = weight_contig.size(1);
    int kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(0) == in_channels, "Weight in_channels mismatch");
    TORCH_CHECK(bias_contig.numel() == out_channels, "Bias size mismatch");

    int out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "Invalid output length");

    auto output = torch::empty({batch, out_channels, out_length}, x_contig.options());

    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    CHECK_CUDNN_ERR(status);

    // Set current stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(cudnn_handle, stream);

    // Input tensor descriptor (N, C, L)
    cudnnTensorDescriptor_t x_desc;
    status = cudnnCreateTensorDescriptor(&x_desc);
    CHECK_CUDNN_ERR(status);
    std::vector<int> x_dims = {batch, in_channels, in_length};
    std::vector<int> x_strides = {in_channels * in_length, in_length, 1};
    status = cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, x_dims.data(), x_strides.data());
    CHECK_CUDNN_ERR(status);

    // Output tensor descriptor
    cudnnTensorDescriptor_t y_desc;
    status = cudnnCreateTensorDescriptor(&y_desc);
    CHECK_CUDNN_ERR(status);
    std::vector<int> y_dims = {batch, out_channels, out_length};
    std::vector<int> y_strides = {out_channels * out_length, out_length, 1};
    status = cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 3, y_dims.data(), y_strides.data());
    CHECK_CUDNN_ERR(status);

    // Filter descriptor: cuDNN expects shape (out_channels, in_channels, kernel_size) for convolution
    // PyTorch ConvTranspose1d weight is (in_channels, out_channels, kernel_size).
    // For cuDNN transposed convolution, we can pass filter with dims: [in_channels, out_channels, kernel_size]
    // and use CUDNN_CROSS_CORRELATION? Actually cuDNN supports transposed convolution via cudnnConvolutionBackwardData,
    // or we can use cudnnConvolutionForward with reversed filter. We'll use cudnnConvolutionBackwardData.
    // Reference: https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionBackwardData
    // For ConvTranspose1d, we treat it as the backward data of a convolution with swapped input/output channels.
    // The weight layout in PyTorch: [in_channels, out_channels, kernel_size]
    // In cuDNN, filter for backward data has dims: [in_channels, out_channels, kernel_size]? Actually cuDNN filter is always [out_channels, in_channels, ...] for forward,
    // but for backward data the filter is still [out_channels, in_channels, ...] where out_channels = number of filters in forward conv.
    // We need to map: forward conv with input channels = out_channels, output channels = in_channels,
    // filter size = kernel_size, stride = stride, padding = padding, output_padding = output_padding.
    // Then backward data gives gradient wrt input, which has shape [batch, in_channels, out_length], matching our desired output channels.
    // But our weight is [in_channels, out_channels, kernel_size], which is transposed version of filter for forward conv with swapped channels.
    // We can create a transposed weight on the fly or set strides accordingly.
    // Easier: create a temporary weight tensor with shape [out_channels, in_channels, kernel_size] and permute dimensions.
    // However, in binding we can allocate a new contiguous tensor with permuted copy via torch::permute(). But cannot use torch operators for compute.
    // Using torch::Tensor::permute is allowed? It returns a view, not compute. We can do weight_contig.permute({1,0,2}).contiguous() to get correct layout.
    // This is a data movement, not computation, and torch::Tensor operations like permute are acceptable as they don't perform arithmetic.
    // But to be safe, we can manually set strides in filter descriptor without copying: cuDNN allows arbitrary strides.
    // We'll set filter dims = {in_channels, out_channels, kernel_size} and strides = {out_channels * kernel_size, kernel_size, 1}.
    // However, cuDNN expects filter dims in order [K, C, ...] where K = number of output feature maps of the forward conv.
    // For backward data, the filter is the same as forward conv filter, with dims [out_channels, in_channels, ...] where out_channels is the number of filters in forward conv.
    // In our case, forward conv would have input channels = out_channels (64), output channels = in_channels (32).
    // So filter dims should be [32, 64, kernel_size]. Our weight already has that shape! Great.
    // So we can directly use weight_contig with shape [32,64,3] as the filter for backward data.

    // Filter descriptor
    cudnnFilterDescriptor_t w_desc;
    status = cudnnCreateFilterDescriptor(&w_desc);
    CHECK_CUDNN_ERR(status);
    std::vector<int> w_dims = {in_channels, out_channels, kernel_size}; // filter dims: [K, C, kernel]
    status = cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, w_dims.data());
    CHECK_CUDNN_ERR(status);

    // Convolution descriptor for backward data
    cudnnConvolutionDescriptor_t conv_desc;
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    CHECK_CUDNN_ERR(status);
    // For 1D, we need to set padding and stride in 1D, but cudnnSetConvolutionNdDescriptor works for N-D.
    // We'll use cudnnSetConvolution2dDescriptor with height=1? Better to use cudnnSetConvolutionNdDescriptor with array size 1.
    int padA[1] = {static_cast<int>(padding)};
    int strideA[1] = {static_cast<int>(stride)};
    int dilationA[1] = {1};
    status = cudnnSetConvolutionNdDescriptor(conv_desc, 1, padA, strideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    CHECK_CUDNN_ERR(status);

    // Output padding for transposed convolution: cudnnConvolutionBackwardData does not directly support output_padding.
    // We need to adjust output size and possibly perform post-cropping.
    // PyTorch formula: out = (in-1)*stride - 2*padding + kernel + output_padding.
    // cuDNN backward data output size (for given input size and conv descriptor) is:
    // out = (in-1)*stride - 2*padding + kernel + dilation*(kernel-1) ... actually formula: out = (in-1)*stride - 2*padding + 1 + (kernel-1)*dilation.
    // With dilation=1, that gives out = (in-1)*stride - 2*padding + kernel.
    // So without output_padding, cuDNN yields length (in_length-1)*stride - 2*padding + kernel_size.
    // We need additional output_padding columns, usually appended at the end. PyTorch adds output_padding extra columns at the end.
    // Approach: Compute backward data with desired output length = (in-1)*stride - 2*padding + kernel_size + output_padding.
    // But cuDNN requires explicit output tensor dimensions; we can set y_desc accordingly, and cuDNN will compute the needed padding? 
    // Actually cuDNN backward data requires the output tensor (dy) to match the input size of the forward conv, and the input tensor (dx) to match output size.
    // Here x is input to conv_transpose (gradient from previous layer), y is output (gradient wrt input).
    // We need to set x_desc for input to backward data (which corresponds to gradient w.r.t. output of forward conv) with shape [batch, out_channels, in_length].
    // Wait, we have x with shape [batch, in_channels, in_length]. That is the input to ConvTranspose1d, which corresponds to output of forward conv with out_channels = in_channels (32).
    // So in backward data, we need to compute gradient wrt input of forward conv, which has shape [batch, out_channels_forward, in_length_forward].
    // Let's define forward conv: input channels = out_channels (64), output channels = in_channels (32), kernel_size, stride, padding.
    // Input length to forward conv = out_length (10240), output length = in_length (5120) (since transposed reduces).
    // Then backward data computes gradient wrt forward input, shape [batch, out_channels, out_length] = [128,64,10240], perfect.
    // The gradient wrt forward output (dy) is our x with shape [batch, in_channels, in_length] = [128,32,5120].
    // So we set:
    //   x_desc: [batch, in_channels, in_length] as dy (gradient wrt forward output)
    //   y_desc: [batch, out_channels, out_length] as dx (gradient wrt forward input)
    //   w_desc: filter for forward conv: shape [in_channels, out_channels, kernel_size] (since forward conv has output channels = in_channels, input channels = out_channels)
    //   conv_desc: stride, padding as given.
    // Nice! This matches exactly.
    // Output_padding: PyTorch's output_padding adds extra length to transposed output. In backward data, we need to produce output length with extra columns.
    // cuDNN backward data output size is determined by input size (dy) and convolution descriptor: out = (in-1)*stride - 2*padding + kernel_size.
    // If we specify y_desc with larger length, cuDNN will likely error or misalign. Instead, we can compute the result without output_padding and then pad with zeros or replicate? PyTorch output_padding pads with zeros at the end.
    // Let's check PyTorch: output_padding adds extra length to output, and it is always zero at the end (no gradient flows back to output_padding).
    // So we can compute y_temp with length (in_length-1)*stride - 2*padding + kernel_size, then allocate final y with + output_padding and copy y_temp into the beginning, leaving zeros at end.
    // But that involves additional copy and zero-initialization. It may be acceptable.
    // Better: use cudnnConvolutionBackwardData with an enlarged output tensor and set the convolution descriptor output padding? cuDNN doesn't have output padding.
    // So we do the two-step approach.

    // Compute base length without output_padding
    int base_out_length = (in_length - 1) * stride - 2 * padding + kernel_size;
    TORCH_CHECK(base_out_length > 0, "Invalid base output length");

    // If output_padding == 0, just compute directly
    if (output_padding == 0) {
        // y_desc already set with out_length = base_out_length
        // We'll set y_desc dimensions to base_out_length
        std::vector<int> y_dims_no_pad = {batch, out_channels, base_out_length};
        std::vector<int> y_strides_no_pad = {out_channels * base_out_length, base_out_length, 1};
        status = cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 3, y_dims_no_pad.data(), y_strides_no_pad.data());
        CHECK_CUDNN_ERR(status);
        // Algorithm selection
        cudnnConvolutionBwdDataAlgo_t algo;
        int requestedAlgoCount = 1;
        int returnedAlgoCount = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perfResults;
        status = cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle, w_desc, x_desc, conv_desc, y_desc, 1, &returnedAlgoCount, &perfResults);
        CHECK_CUDNN_ERR(status);
        algo = perfResults.algo;
        // Get workspace size
        size_t workspace_size = 0;
        status = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle, w_desc, x_desc, conv_desc, y_desc, algo, &workspace_size);
        CHECK_CUDNN_ERR(status);
        // Allocate workspace if needed
        void* workspace = nullptr;
        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
            if (cudaGetLastError() != cudaSuccess) {
                TORCH_CHECK(false, "cudaMalloc failed for workspace");
            }
        }
        // Run
        status = cudnnConvolutionBackwardData(cudnn_handle, &alpha, w_desc, weight_contig.data_ptr<float>(),
                                             x_desc, x_contig.data_ptr<float>(),
                                             conv_desc, algo, workspace, workspace_size,
                                             &beta, y_desc, output.data_ptr<float>());
        CHECK_CUDNN_ERR(status);
        if (workspace) cudaFree(workspace);
    } else {
        // Compute without output_padding, then copy and pad zeros
        auto output_base = torch::empty({batch, out_channels, base_out_length}, x_contig.options());
        // y_desc for base
        std::vector<int> y_dims_base = {batch, out_channels, base_out_length};
        std::vector<int> y_strides_base = {out_channels * base_out_length, base_out_length, 1};
        status = cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 3, y_dims_base.data(), y_strides_base.data());
        CHECK_CUDNN_ERR(status);
        // Algorithm selection
        cudnnConvolutionBwdDataAlgo_t algo;
        int requestedAlgoCount = 1;
        int returnedAlgoCount = 0;
        cudnnConvolutionBwdDataAlgoPerf_t perfResults;
        status = cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle, w_desc, x_desc, conv_desc, y_desc, 1, &returnedAlgoCount, &perfResults);
        CHECK_CUDNN_ERR(status);
        algo = perfResults.algo;
        size_t workspace_size = 0;
        status = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle, w_desc, x_desc, conv_desc, y_desc, algo, &workspace_size);
        CHECK_CUDNN_ERR(status);
        void* workspace = nullptr;
        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
        }
        status = cudnnConvolutionBackwardData(cudnn_handle, &alpha, w_desc, weight_contig.data_ptr<float>(),
                                             x_desc, x_contig.data_ptr<float>(),
                                             conv_desc, algo, workspace, workspace_size,
                                             &beta, y_desc, output_base.data_ptr<float>());
        CHECK_CUDNN_ERR(status);
        if (workspace) cudaFree(workspace);
        // Copy base output to final output with zero padding at end
        // output is already allocated with out_length; initialize to zero
        output.zero_(); // This zeroes entire tensor, but we then copy base.
        // Copy base data into first base_out_length columns for each batch and channel.
        // Use cudaMemcpyAsync 2D? We can use a simple kernel or torch::cat? torch::cat is a torch op, not allowed.
        // We can use cudaMemcpy2DAsync for each channel: copy out_channels * batch times.
        // Or allocate a single contiguous copy of base to output's storage beginning.
        // Since output and output_base are both contiguous, we can copy the entire base tensor into the front of output using cudaMemcpyAsync.
        // output has numel = batch*out_channels*out_length, output_base numel = batch*out_channels*base_out_length.
        // We need to copy base into the first part of output, leaving the rest zeros.
        // We can do cudaMemcpyAsync(output.data_ptr(), output_base.data_ptr(), base_numel*sizeof(float), cudaMemcpyDeviceToDevice, stream);
        size_t base_numel = batch * out_channels * base_out_length;
        cudaMemcpyAsync(output.data_ptr<float>(), output_base.data_ptr<float>(), base_numel * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        CHECK_CUDA_ERR();
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("hardswish_forward", &hardswish_forward, "Hardswish forward", pybind11::arg("x"));
    m.def("softplus_forward", &softplus_forward, "Softplus forward", pybind11::arg("x"));
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward, "ConvTranspose1d forward",
          pybind11::arg("x"), pybind11::arg("weight"), pybind11::arg("bias"),
          pybind11::arg("stride"), pybind11::arg("padding"), pybind11::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
