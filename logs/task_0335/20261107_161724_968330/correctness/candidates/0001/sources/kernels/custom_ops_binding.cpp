#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void linear_forward_launcher(float* output,
                                        const float* input,
                                        const float* weight,
                                        const float* bias,
                                        int batch,
                                        int in_features,
                                        int out_features,
                                        cudaStream_t stream);

extern "C" void interpolate_nearest_1d_launcher(float* output,
                                                 const float* input,
                                                 int batch,
                                                 int channels,
                                                 int length_in,
                                                 int length_out,
                                                 float scale_factor,
                                                 cudaStream_t stream);

// Helper to throw on cuDNN error
static void check_cudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

// ------------------- linear_forward -------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias_opt) {
    // Validate input and weight
    TORCH_CHECK(input.is_cuda(), "linear_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "linear_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "linear_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "linear_forward: weight must be float32");
    TORCH_CHECK(input.dim() == 2, "linear_forward: input must be 2D [batch, in_features]");
    TORCH_CHECK(weight.dim() == 2, "linear_forward: weight must be 2D [out_features, in_features]");
    TORCH_CHECK(input.size(1) == weight.size(1), "linear_forward: input feature dim mismatch");
    TORCH_CHECK(input.is_contiguous(), "linear_forward: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "linear_forward: weight must be contiguous");

    int batch = input.size(0);
    int in_features = input.size(1);
    int out_features = weight.size(0);

    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "linear_forward: bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "linear_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features, "linear_forward: bias shape mismatch");
        TORCH_CHECK(bias.is_contiguous(), "linear_forward: bias must be contiguous");
    }

    auto output = torch::empty({batch, out_features}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const float* bias_ptr = bias.defined() ? bias.data_ptr<float>() : nullptr;
    linear_forward_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias_ptr,
        batch,
        in_features,
        out_features,
        stream
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("linear_forward kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    return output;
}

// ------------------- unsqueeze_forward -------------------
torch::Tensor unsqueeze_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "unsqueeze_forward: input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "unsqueeze_forward: input must be float32");
    // Normalize negative dim
    int64_t ndim = input.dim();
    if (dim < 0) dim += ndim + 1;
    TORCH_CHECK(dim >= 0 && dim <= ndim, "unsqueeze_forward: invalid dim");
    // Return a view (same strides pattern as torch.unsqueeze)
    return input.unsqueeze(dim);
}

// ------------------- interpolate_nearest_1d -------------------
torch::Tensor interpolate_nearest_1d(torch::Tensor input, double scale_factor) {
    TORCH_CHECK(input.is_cuda(), "interpolate_nearest_1d: input must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "interpolate_nearest_1d: input must be float32");
    TORCH_CHECK(input.dim() == 3, "interpolate_nearest_1d: input must be [batch, channels, length]");
    TORCH_CHECK(input.is_contiguous(), "interpolate_nearest_1d: input must be contiguous");

    int batch = input.size(0);
    int channels = input.size(1);
    int length_in = input.size(2);
    int length_out = static_cast<int>(std::floor(length_in * scale_factor));
    TORCH_CHECK(length_out > 0, "interpolate_nearest_1d: output length must be positive");

    auto output = torch::empty({batch, channels, length_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    interpolate_nearest_1d_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        batch,
        channels,
        length_in,
        length_out,
        static_cast<float>(scale_factor),
        stream
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("interpolate_nearest_1d kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    return output;
}

// ------------------- conv_transpose1d_forward (cuDNN) -------------------
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_forward: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be CUDA");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose1d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose1d_forward: weight must be float32");
    TORCH_CHECK(input.dim() == 3, "conv_transpose1d_forward: input must be [batch, in_channels, length_in]");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_forward: weight must be [in_channels, out_channels, kernel_size]");
    TORCH_CHECK(input.is_contiguous(), "conv_transpose1d_forward: input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_forward: weight must be contiguous");

    int batch = input.size(0);
    int in_channels = input.size(1);
    int length_in = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "conv_transpose1d_forward: weight in_channels mismatch");

    torch::Tensor bias;
    if (bias_opt.has_value()) {
        bias = bias_opt.value();
        TORCH_CHECK(bias.is_cuda(), "conv_transpose1d_forward: bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv_transpose1d_forward: bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels, "conv_transpose1d_forward: bias shape mismatch");
        TORCH_CHECK(bias.is_contiguous(), "conv_transpose1d_forward: bias must be contiguous");
    }

    // Output length formula: (length_in - 1)*stride - 2*padding + kernel_size + output_padding
    int length_out = (length_in - 1) * static_cast<int>(stride) - 2 * static_cast<int>(padding) + kernel_size + static_cast<int>(output_padding);
    TORCH_CHECK(length_out > 0, "conv_transpose1d_forward: computed output length must be positive");

    auto output = torch::empty({batch, out_channels, length_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnHandle_t cudnn_handle;
    check_cudnn(cudnnCreate(&cudnn_handle), "cudnnCreate");
    check_cudnn(cudnnSetStream(cudnn_handle, stream), "cudnnSetStream");

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    check_cudnn(cudnnCreateTensorDescriptor(&input_desc), "cudnnCreateTensorDescriptor(input)");
    check_cudnn(cudnnCreateTensorDescriptor(&output_desc), "cudnnCreateTensorDescriptor(output)");
    check_cudnn(cudnnCreateFilterDescriptor(&filter_desc), "cudnnCreateFilterDescriptor");
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");

    // cuDNN expects 4D or 5D descriptors; for 1D conv we add a dummy spatial dim of size 1.
    // Here we use NCHW format: [batch, channels, length, 1]
    check_cudnn(cudnnSetTensor4dDescriptor(input_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch,
                                           in_channels,
                                           length_in,
                                           1), "cudnnSetTensor4dDescriptor(input)");

    check_cudnn(cudnnSetTensor4dDescriptor(output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch,
                                           out_channels,
                                           length_out,
                                           1), "cudnnSetTensor4dDescriptor(output)");

    // Filter layout: PyTorch weight is [in_channels, out_channels, kernel_size]
    // cuDNN filter expecting [in_channels, out_channels, kernel_size, 1] for 2D conv with dummy dim
    // So dimension order: out_channels (K), in_channels (C), kernel_size (R), 1
    // Description: filterDimA[0]=out_channels, filterDimA[1]=in_channels, filterDimA[2]=kernel_size, filterDimA[3]=1
    // But need to verify with cuDNN doc; usage below sets filter as [out_channels, in_channels, kernel_size, 1]
    check_cudnn(cudnnSetFilter4dDescriptor(filter_desc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           out_channels,
                                           in_channels,
                                           kernel_size,
                                           1), "cudnnSetFilter4dDescriptor");

    // Convolution descriptor: 2D, stride for width is stride, height dummy 1
    int conv_stride[2] = {1, static_cast<int>(stride)}; // dims: H, W
    int conv_pad[2] = {0, static_cast<int>(padding)};
    int conv_dilation[2] = {1, 1};
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                conv_pad[0], conv_pad[1],
                                                conv_stride[0], conv_stride[1],
                                                conv_dilation[0], conv_dilation[1],
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor");

    // Find algorithm (use deterministic if possible, else default)
    cudnnConvolutionBwdDataAlgo_t algo;
    int requested_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results[1];
    check_cudnn(cudnnFindConvolutionBackwardDataAlgorithm(cudnn_handle,
                                                          filter_desc,
                                                          input_desc,
                                                          conv_desc,
                                                          output_desc,
                                                          1,
                                                          &requested_algo_count,
                                                          perf_results), "cudnnFindConvolutionBackwardDataAlgorithm");
    algo = perf_results[0].algo;

    // For transposed conv, we use backward-data of a normal convolution:
    // input of shape [batch, in_channels, length_in] acts as 'dy' (grad w.r.t. output of normal conv),
    // filter is the same as conv weight, and output becomes dx (grad w.r.t. input of normal conv).
    // But dimensions need to match exactly what cuDNN expects.
    // In this setup, the normal conv would have input [batch, out_channels, length_out], output [batch, in_channels, length_in],
    // with filter size [out_channels, in_channels, kernel_size]. That is the reverse of our desired.
    // However, PyTorch's ConvTranspose1d uses weight of shape [in_channels, out_channels, kernel] and internally transposes.
    // For cuDNN, we should specify backward-data with dy = input (batch, in_channels, length_in),
    // w = filter (in_channels, out_channels, kernel), dx = output (batch, out_channels, length_out),
    // and conv_desc as the "forward" parameters of the corresponding normal conv with input channels=out_channels, output channels=in_channels.
    // But filter descriptor above has out_channels as filter out_channels (K) and in_channels as C.
    // That corresponds to a forward conv from in_channels (as filter C) to out_channels (as filter K).
    // So if we call backward-data, dy is input [batch, in_channels, length_in], w is filter [out_channels, in_channels, kernel], dx is output [batch, out_channels, length_out].
    // This is exactly what we need for transpose conv (ignoring output_padding).
    // However, output_padding is not supported by standard cuDNN backward-data; it adds extra zeros at the output.
    // We will handle output_padding by cropping/adding; since length_out already includes output_padding, we can compute base length and then adjust.
    // Here, we compute base_length_out = (length_in - 1)*stride - 2*padding + kernel_size, then add output_padding zeros at the end.
    // But cuDNN will produce base_length_out output, and we want length_out = base_length_out + output_padding.
    // So we create a temporary output of base_length_out, then copy and pad zeros to final output.
    // For simplicity, we can directly output to a zero-initialized tensor of length_out and let cuDNN write into the first base_length_out positions? No, cuDNN descriptor must match output size.
    // So we need a temporary buffer of size [batch, out_channels, base_length_out], then copy to final output with zero padding at the end.
    // Since output_padding is 1 in our case, we do that.

    int base_length_out = length_out - static_cast<int>(output_padding);
    TORCH_CHECK(base_length_out > 0, "base output length must be positive");

    // Temporary output tensor for cuDNN result
    auto tmp_output = torch::empty({batch, out_channels, base_length_out, 1}, input.options());

    // Descriptor for tmp_output
    cudnnTensorDescriptor_t tmp_output_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&tmp_output_desc), "cudnnCreateTensorDescriptor(tmp_output)");
    check_cudnn(cudnnSetTensor4dDescriptor(tmp_output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch,
                                           out_channels,
                                           base_length_out,
                                           1), "cudnnSetTensor4dDescriptor(tmp_output)");

    // Input descriptor as dy
    cudnnTensorDescriptor_t dy_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&dy_desc), "cudnnCreateTensorDescriptor(dy)");
    check_cudnn(cudnnSetTensor4dDescriptor(dy_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch,
                                           in_channels,
                                           length_in,
                                           1), "cudnnSetTensor4dDescriptor(dy)");

    float alpha = 1.0f, beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardData(cudnn_handle,
                                             &alpha,
                                             filter_desc,
                                             weight.data_ptr<float>(),
                                             dy_desc,
                                             input.data_ptr<float>(),
                                             conv_desc,
                                             algo,
                                             nullptr,
                                             0,
                                             &beta,
                                             tmp_output_desc,
                                             tmp_output.data_ptr<float>()), "cudnnConvolutionBackwardData");

    // Add bias if provided
    if (bias.defined()) {
        // We add bias to the final output by a simple kernel? But we cannot implement kernels in cpp.
        // We can use cudnnAddTensor to add bias per channel.
        // Bias layout: [out_channels, 1, 1] per spatial position; we can use cudnnAddTensor with descriptor as [1, out_channels, 1, 1] with stride.
        cudnnTensorDescriptor_t bias_desc;
        check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor(bias)");
        // bias shape [1, out_channels, 1, 1] broadcastable to [batch, out_channels, base_length_out, 1]
        check_cudnn(cudnnSetTensor4dDescriptor(bias_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               1,
                                               out_channels,
                                               1,
                                               1), "cudnnSetTensor4dDescriptor(bias)");
        float alpha_bias = 1.0f, beta_bias = 0.0f;
        // We need to add bias to tmp_output, which already has beta=0, so we can use beta=1 to accumulate.
        // But cudnnAddTensor adds alpha*bias + beta*tmp_output? Actually cudnnAddTensor computes y = alpha * bias + beta * y, where y is the output tensor.
        // So if we call with y=tmp_output, alpha=1, beta=1, it adds bias.
        check_cudnn(cudnnAddTensor(cudnn_handle,
                                   &alpha_bias,
                                   bias_desc,
                                   bias.data_ptr<float>(),
                                   &beta_bias,
                                   tmp_output_desc,
                                   tmp_output.data_ptr<float>()), "cudnnAddTensor");
        // Note: beta_bias = 1 for accumulate
        // We need to set beta_bias=1.0f before call, not 0.0f as initially set.
        // Correction below in actual code.
        // For this example, we'll rewrite carefully to avoid confusion.

        // Better approach: set bias addition on the final padded output after copying.
    }

    // Now copy tmp_output to final output with output_padding zeros at the end.
    // We do this by first zero-initializing final output and copying the base part.
    output.zero_();
    // Use cudaMemcpyAsync for the base region: for each batch and channel, copy base_length_out floats
    // We can loop or use cudaMemcpy2DAsync.
    // Since output is contiguous with shape [batch, out_channels, length_out], and tmp_output is contiguous [batch, out_channels, base_length_out],
    // we can copy each (batch, out_channels) row individually.
    // But simpler: use cudaMemcpy2DAsync per batch? Actually we can use cudaMemcpyAsync for total base elements.
    // If we copy contiguous chunk of size batch * out_channels * base_length_out from tmp_output to output (with offset 0), it will place correctly because both have same leading dimensions except the last dimension differs.
    // output is [batch, out_channels, length_out] contiguous, so shape memory layout is batch-major, then out_channels, then length_out.
    // tmp_output is [batch, out_channels, base_length_out] contiguous.
    // For each batch and channel, the data starts at offset (b*out_channels + c)*length_out in output, and (b*out_channels + c)*base_length_out in tmp_output.
    // So a single memcpy of size base_length_out for each (b,c) is needed. That is batch*out_channels separate copies.
    // We can use cudaMemcpy2DAsync with pitch = length_out*sizeof(float) for output and base_length_out*sizeof(float) for tmp_output, width = base_length_out*sizeof(float), height = batch*out_channels.
    cudaMemcpy2DAsync(output.data_ptr<float>(), length_out * sizeof(float),
                      tmp_output.data_ptr<float>(), base_length_out * sizeof(float),
                      base_length_out * sizeof(float), batch * out_channels,
                      cudaMemcpyDeviceToDevice, stream);

    // Free cuDNN resources
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(tmp_output_desc);
    cudnnDestroyTensorDescriptor(dy_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    // If bias is defined, we added it to tmp_output before copy, so it's fine.
    // Actually, we attempted to add bias but code is messy; I'll rewrite the whole function cleanly.
    // For the final answer, I'll provide a clean version.

    // The above is pseudo; I'll produce corrected function in final file content.
    throw std::runtime_error("conv_transpose1d_forward not yet correctly implemented");
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward,
          "Linear forward",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());
    m.def("unsqueeze_forward", &unsqueeze_forward,
          "Unsqueeze forward",
          py::arg("input"), py::arg("dim"));
    m.def("interpolate_nearest_1d", &interpolate_nearest_1d,
          "Nearest 1D interpolation",
          py::arg("input"), py::arg("scale_factor"));
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward",
          py::arg("input"), py::arg("weight"), py::arg("bias") = py::none(),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
