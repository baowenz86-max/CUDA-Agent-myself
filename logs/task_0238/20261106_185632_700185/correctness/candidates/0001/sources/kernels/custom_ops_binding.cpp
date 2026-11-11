#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// External declarations from custom_ops.cu
extern "C" void clamp_log2_launcher(const float* x, float* y, int64_t numel, int64_t ndim,
                                    const int64_t* shape, const int64_t* strides, float min_val, cudaStream_t stream);
extern "C" void mse_loss_launcher(const float* x, const float* target, float* output, int64_t numel,
                                  int64_t ndim, const int64_t* x_shape, const int64_t* x_strides,
                                  const int64_t* target_shape, const int64_t* target_strides, cudaStream_t stream);

// Helper to check cuDNN status
#define CHECK_CUDNN(expr) \
    do { \
        cudnnStatus_t status = (expr); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
        } \
    } while (0)

// ConvTranspose1d forward using cuDNN
static torch::Tensor conv_transpose1d_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias,
                                              int64_t in_channels, int64_t out_channels, int64_t kernel_size,
                                              int64_t stride, int64_t padding, int64_t output_padding) {
    // Check inputs
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (C_in, C_out, kernel_size)");
    TORCH_CHECK(weight.size(0) == in_channels, "weight first dim must equal in_channels");
    TORCH_CHECK(weight.size(1) == out_channels, "weight second dim must equal out_channels");
    TORCH_CHECK(weight.size(2) == kernel_size, "weight third dim must equal kernel_size");

    bool has_bias = bias.defined();
    if (has_bias) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_channels, "bias must be 1D of size out_channels");
    }

    int64_t N = x.size(0);
    int64_t L_in = x.size(2);
    // L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "invalid output length");

    // Ensure tensors are contiguous as required by cuDNN
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = has_bias ? bias.contiguous() : torch::Tensor();

    // Allocate output tensor
    auto output = torch::empty({N, out_channels, L_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));

    // cuDNN descriptor for input/output tensors (1D convolution requires 2D spatial? We use CUDNN_TENSOR_NCHW with dummy H dimension)
    // Actually cuDNN supports 1D convolution via 2D with height=1. We'll use CUDNN_TENSOR_NCHW with H=1.
    cudnnTensorDescriptor_t x_desc, y_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, in_channels, 1, L_in));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, out_channels, 1, L_out));

    // Filter descriptor for transposed convolution (weight shape [C_in, C_out, kernel_size])
    // cuDNN expects filter shape [C_out, C_in, filter_h, filter_w] for convolution, but for deconvolution
    // the filter is [C_in, C_out, k_h, k_w]? We need to check cuDNN documentation. In cuDNN,
    // cudnnSetFilter4dDescriptor for backward data (deconvolution) expects filter dims [C_in, C_out, kh, kw]
    // if using CUDNN_TENSOR_NCHW. So we'll set filter as (in_channels, out_channels, 1, kernel_size).
    cudnnFilterDescriptor_t filter_desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           in_channels, out_channels, 1, kernel_size));

    // Convolution descriptor for transposed convolution (backward data)
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding, padding,          // pad_h, pad_w (1D: pad_h=0? Actually padding is vertical? We set pad_h=0, pad_w=padding)
                                                stride, stride,            // stride_h, stride_w (1D: stride_h=1? Actually we treat as 2D with dummy height)
                                                1, 1,                      // dilation_h, dilation_w
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    // For transposed convolution, cuDNN uses "backward data" algorithm.
    // We need to set the convolution mode correctly. The API cudnnConvolutionBackwardData is what we need.
    // But for simplicity, we can use cudnnConvolutionForward with the filter flipped? No, better to use cudnnConvolutionBackwardData.
    // However, cuDNN's backward data computes dx from dy and filter, which matches deconvolution.
    // The formula for transposed convolution output L_out = (L_in-1)*stride -2*padding + kernel_size + output_padding.
    // cuDNN's backward data for a convolution with stride s, padding p, kernel k gives output length L_out = (L_in-1)*s - 2*p + k.
    // output_padding is handled by adjusting the output dimension after computation; but it should be included in the output tensor size.
    // Actually output_padding adds extra zeros at the end, which we can implement by allocating output with L_out and zeroing the extra positions.
    // Our computed L_out includes output_padding. cuDNN will produce length L_out_no_outpad = (L_in-1)*s -2*p + k. We need to zero the last output_padding positions.
    // However, output_padding is typically 0 or 1, and it only affects the last dimension.
    // To keep it simple, we can use cudnnConvolutionBackwardData to compute the main part, then zero the extra output_padding elements.

    // Determine alpha/beta for bias handling
    float alpha = 1.0f, beta = 0.0f;

    // Workspace
    size_t workspace_size = 0;
    void* workspace = nullptr;

    // Algorithm selection
    cudnnConvolutionBwdDataAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        cudnn_handle, filter_desc, y_desc, conv_desc, x_desc,
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle, filter_desc, y_desc, conv_desc, x_desc, algo, &workspace_size));
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Perform convolution backward data:
    // dy is the "output gradient" which corresponds to the transposed conv input x? Actually in cuDNN backward data:
    // dx = conv_backward_data(dy, w, conv_desc, x_desc) where dy has shape of forward output (N, C_out, L_out_wo_pad)
    // But here our x is the input to transposed conv, which acts as dy in backward data. So we pass x as dy (using x_desc?). Let's clarify:
    // For transposed conv, we want y = deconv(x, w). In cuDNN, that's equivalent to computing dx given dy=x, filter=w, and conv_desc describing the forward conv.
    // So we need to set dy descriptor to x_desc (shape N, C_in, L_in), and dx descriptor to y_desc (shape N, C_out, L_out_no_outpad).
    // filter_desc should be for the forward conv, which is [C_out, C_in, kernel_size] (if C_out is output channels of forward, but in transposed weight is [C_in, C_out, k] so we need to permute? Actually cuDNN expects filter for backward data as [C_out, C_in, kh, kw] if forward conv filter is [C_out, C_in, kh, kw]. But our weight is [C_in, C_out, k] which is transposed. So we need to either permute weight or treat C_in and C_out swapped in descriptors. We can avoid permutation by setting filter_desc as [C_in, C_out, 1, k] and use appropriate conv descriptors. Wait, in cudnnSetFilter4dDescriptor, the first two dimensions are K and C (for forward), but for backward data, the filter is the same as forward filter? Let's recall: cudnnConvolutionBackwardData takes a filter descriptor that describes the filter used in the forward convolution (shape [K, C, R, S] where K is number of output channels, C input channels). For transposed conv, weight is [C_in, C_out, k] which is the transpose of forward filter [C_out, C_in, k]. So we cannot directly use it without transposing. We can either create a transposed filter copy or use cudnnSetFilter4dDescriptor with format CUDNN_TENSOR_NHWC? That may allow different ordering? Not simple.
    // Alternative: use cudnnConvolutionForward with the input/output dimensions swapped and filter flipped.
    // Given complexity, maybe we should implement transposed convolution using cudnnConvolutionForward with appropriate settings:
    // A transposed convolution with stride s, padding p, kernel k can be implemented as a standard convolution with stride 1, padding k-p-1, and dilation s, but only if s>1? Actually it's equivalent to a fractionally strided convolution.
    // But the cuDNN guide suggests using cudnnConvolutionBackwardData for deconvolution. We must permute weight to [C_out, C_in, k] and then use filter descriptor [C_out, C_in, 1, k].
    // Since performance is not critical for correctness (we have small weights), we can create a temporary permuted weight on device and then use it.
    // We'll do that to ensure correctness.

    // Create permuted weight tensor (C_out, C_in, 1, k) from weight_contig (C_in, C_out, k)
    auto weight_perm = torch::empty({out_channels, in_channels, 1, kernel_size}, weight.options());
    // Use a simple CUDA kernel or torch operations? We cannot use torch ops. We can use a custom kernel but that would require adding to .cu. Since this binding is supposed to only use cuDNN, we can use cudaMemcpy and loops? But we can't launch kernels from .cpp. We need to ensure no torch ops. We can allocate device memory and use a small CUDA kernel, but we are not allowed to define kernels in .cpp. We could use cudaMemcpy2D if the data is contiguous, but weight_contig is contiguous and shape (C_in, C_out, k). We need to transpose the first two dims. This is not trivial without a kernel.
    // Since the weight is small (3*16*3=144 floats), we can copy to CPU, permute, and copy back. That's acceptable for small weights and avoids kernels. But we cannot use torch CPU ops, only raw C++. So we can do:
    // 1. Copy weight_contig to CPU (using cudaMemcpy DeviceToHost)
    // 2. Create host array of permuted weights
    // 3. Copy permuted to device
    // This is allowed since it's not using torch operators, just cudaMemcpy and plain C++ loops.

    std::vector<float> h_weight(weight_contig.numel());
    cudaMemcpy(h_weight.data(), weight_contig.data_ptr<float>(), weight_contig.numel() * sizeof(float), cudaMemcpyDeviceToHost);

    // Permute: weight_contig layout: [C_in][C_out][k] -> permuted: [C_out][C_in][1][k]
    int64_t C_in = in_channels, C_out = out_channels, K = kernel_size;
    std::vector<float> h_perm_weight(C_out * C_in * 1 * K);
    for (int64_t ci = 0; ci < C_in; ++ci) {
        for (int64_t co = 0; co < C_out; ++co) {
            for (int64_t k = 0; k < K; ++k) {
                // weight_contig index: ci*C_out*K + co*K + k
                // permuted index: co*C_in*1*K + ci*1*K + 0*K + k = co*C_in*K + ci*K + k
                h_perm_weight[co * C_in * K + ci * K + k] = h_weight[ci * C_out * K + co * K + k];
            }
        }
    }

    // Allocate device memory for permuted weight
    float* d_perm_weight;
    cudaMalloc(&d_perm_weight, h_perm_weight.size() * sizeof(float));
    cudaMemcpy(d_perm_weight, h_perm_weight.data(), h_perm_weight.size() * sizeof(float), cudaMemcpyHostToDevice);

    // Set filter descriptor using permuted weight format [C_out, C_in, 1, K]
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           C_out, C_in, 1, K));

    // Update convolution descriptor for backward data; we need to specify output_padding handling.
    // cuDNN backward data output length is (L_in - 1)*stride - 2*padding + K, without output_padding.
    // We'll compute that as L_out_base and then later add zeros for output_padding extra elements.
    int64_t L_out_base = (L_in - 1) * stride - 2 * padding + kernel_size; // without output_padding
    // Allocate output_base of that length, then create final output with L_out and zero the last output_padding elements.
    torch::Tensor output_base = torch::empty({N, out_channels, L_out_base}, x.options());
    torch::Tensor output_final = output; // already allocated with L_out

    // Zero output_final
    cudaMemsetAsync(output_final.data_ptr<float>(), 0, output_final.numel() * sizeof(float), stream);

    // Perform convolution backward data: dy (input x) -> dx (output_base)
    // x_desc is for dy (N, in_channels, 1, L_in)
    // y_desc is for dx (N, out_channels, 1, L_out_base)
    // We need a descriptor for output_base.
    cudnnTensorDescriptor_t dx_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&dx_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, out_channels, 1, L_out_base));

    // Set beta to 0 for output_base (we don't accumulate)
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        d_perm_weight,
        x_desc,
        x_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace,
        workspace_size,
        &beta,
        dx_desc,
        output_base.data_ptr<float>()));

    // Now output_base contains the convolution result without output_padding extra zeros.
    // Copy output_base into output_final at appropriate positions. Since output_padding is usually 0 or 1, we can copy the entire base and then zero the last output_padding elements (already done by memset).
    // But we need to copy output_base to output_final. Since output_final is larger only in last dimension, we can do a strided copy for each sample and channel.
    // We'll use cudaMemcpy2DAsync for each N, C pair, copying L_out_base floats from output_base to output_final, with row length L_out_base and pitch L_out_base*sizeof(float), and destination pitch L_out*sizeof(float).
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < out_channels; ++c) {
            const float* src = output_base.data_ptr<float>() + n * out_channels * L_out_base + c * L_out_base;
            float* dst = output_final.data_ptr<float>() + n * out_channels * L_out + c * L_out;
            cudaMemcpy2DAsync(dst, L_out * sizeof(float), src, L_out_base * sizeof(float),
                              L_out_base * sizeof(float), 1, cudaMemcpyDeviceToDevice, stream);
        }
    }

    // Add bias if present
    if (has_bias) {
        // Bias addition: y[n,c,l] += bias[c]. We need a kernel but we cannot define one in .cpp. However, we can use cudnnAddTensor.
        // Create bias descriptor as tensor of shape (1, out_channels, 1, 1) and use cudnnAddTensor with beta=1.
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                               1, out_channels, 1, 1));
        float one = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(cudnn_handle, &one, bias_desc, bias_contig.data_ptr<float>(),
                                   &one, y_desc, output_final.data_ptr<float>()));
    }

    // Cleanup
    cudaFree(d_perm_weight);
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(dx_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output_final;
}

// clamp_log2 forward wrapper
torch::Tensor clamp_log2_forward(torch::Tensor x, double min_val = 1e-7) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto output = torch::empty_like(x);

    // Get shape and strides as host vectors
    auto sizes = x.sizes();
    auto strides = x.strides();
    int64_t ndim = x.dim();
    std::vector<int64_t> shape(sizes.begin(), sizes.end());
    std::vector<int64_t> stride_vec(strides.begin(), strides.end());

    // Copy shape/strides to device
    int64_t* d_shape;
    int64_t* d_strides;
    cudaMalloc(&d_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_shape, shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_strides, stride_vec.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    clamp_log2_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        ndim,
        d_shape,
        d_strides,
        static_cast<float>(min_val),
        stream
    );

    cudaFree(d_shape);
    cudaFree(d_strides);

    return output;
}

// mse_loss forward wrapper
torch::Tensor mse_loss_forward(torch::Tensor x, torch::Tensor target) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(target.dtype() == torch::kFloat32, "target must be float32");
    TORCH_CHECK(x.sizes() == target.sizes(), "x and target must have same shape");

    auto output = torch::empty({}, x.options()); // scalar tensor

    // Get shapes/strides
    auto x_sizes = x.sizes();
    auto x_strides = x.strides();
    auto tgt_sizes = target.sizes();
    auto tgt_strides = target.strides();
    int64_t ndim = x.dim();
    std::vector<int64_t> x_shape(x_sizes.begin(), x_sizes.end());
    std::vector<int64_t> x_stride(x_strides.begin(), x_strides.end());
    std::vector<int64_t> tgt_shape(tgt_sizes.begin(), tgt_sizes.end());
    std::vector<int64_t> tgt_stride(tgt_strides.begin(), tgt_strides.end());

    // Copy to device
    int64_t* d_x_shape;
    int64_t* d_x_strides;
    int64_t* d_tgt_shape;
    int64_t* d_tgt_strides;
    cudaMalloc(&d_x_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_x_strides, ndim * sizeof(int64_t));
    cudaMalloc(&d_tgt_shape, ndim * sizeof(int64_t));
    cudaMalloc(&d_tgt_strides, ndim * sizeof(int64_t));
    cudaMemcpy(d_x_shape, x_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x_strides, x_stride.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt_shape, tgt_shape.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tgt_strides, tgt_stride.data(), ndim * sizeof(int64_t), cudaMemcpyHostToDevice);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    mse_loss_launcher(
        x.data_ptr<float>(),
        target.data_ptr<float>(),
        output.data_ptr<float>(),
        x.numel(),
        ndim,
        d_x_shape,
        d_x_strides,
        d_tgt_shape,
        d_tgt_strides,
        stream
    );

    cudaFree(d_x_shape);
    cudaFree(d_x_strides);
    cudaFree(d_tgt_shape);
    cudaFree(d_tgt_strides);

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("in_channels"), py::arg("out_channels"), py::arg("kernel_size"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("clamp_log2_forward", &clamp_log2_forward,
          "clamp and log2 forward",
          py::arg("x"), py::arg("min") = 1e-7);
    m.def("mse_loss_forward", &mse_loss_forward,
          "MSE loss forward",
          py::arg("x"), py::arg("target"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
