#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Forward declaration of softmax launcher from custom_ops.cu
extern "C" void softmax_dim1_launcher(
    const float* input,
    float* output,
    int N, int C, int L,
    long long in_stride_n, long long in_stride_c, long long in_stride_l,
    long long out_stride_n, long long out_stride_c, long long out_stride_l,
    cudaStream_t stream);

// Helper to get current cudnn handle (thread-local)
static cudnnHandle_t& get_cudnn_handle() {
    static thread_local cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
    }
    return handle;
}

// cuDNN error checking macro
#define CUDNN_CHECK(expr) \
    do { \
        cudnnStatus_t status = (expr); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            throw std::runtime_error(std::string("cuDNN error: ") + cudnnGetErrorString(status)); \
        } \
    } while (0)

// ConvTranspose1d using cuDNN
// input: [N, C_in, L_in] contiguous
// weight: [C_in, C_out, kernel_size] contiguous
// bias: [C_out] contiguous or None
// stride, padding, output_padding: ints
// output: [N, C_out, L_out] where L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding
torch::Tensor conv_transpose1d_ext(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,  // may be undefined for no bias
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Check devices
    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_ext: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_ext: weight must be a CUDA tensor");
    if (bias.defined() && !bias.is_cuda()) {
        TORCH_CHECK(false, "conv_transpose1d_ext: bias must be a CUDA tensor");
    }

    // Check dtypes
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv_transpose1d_ext: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv_transpose1d_ext: weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "conv_transpose1d_ext: bias must be float32");
    }

    // Check dimensions
    TORCH_CHECK(input.dim() == 3, "conv_transpose1d_ext: input must be 3D (N, C_in, L_in)");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_ext: weight must be 3D (C_in, C_out, kernel_size)");

    int64_t N = input.size(0);
    int64_t C_in = input.size(1);
    int64_t L_in = input.size(2);
    int64_t C_out = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "conv_transpose1d_ext: weight shape mismatch: weight.size(0) must equal input channels");
    if (bias.defined()) {
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C_out, "conv_transpose1d_ext: bias must be 1D with size C_out");
    }

    // Compute output length
    int64_t L_out = (L_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(L_out > 0, "conv_transpose1d_ext: computed output length is non-positive");

    // Ensure contiguous tensors for cuDNN
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.defined() ? bias.contiguous() : torch::Tensor();

    // Allocate output tensor (contiguous)
    auto output = torch::empty({N, C_out, L_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnHandle_t handle = get_cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    // Since cuDNN does not support 1D convolutions directly, we use 2D descriptors with height=1
    // Input: [N, C_in, 1, L_in]
    // Output: [N, C_out, 1, L_out]
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, 1, L_in));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, 1, L_out));

    // Filter descriptor for transposed convolution: shape should be [C_in, C_out, 1, kernel_size]
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_in, C_out, 1, kernel_size));

    // Convolution descriptor: use "backward data" mode for transposed conv.
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding,           // vertical padding (applied to both sides)
        padding,           // horizontal padding
        stride,            // vertical stride
        stride,            // horizontal stride
        1, 1,              // dilation
        CUDNN_CROSS_CORRELATION, // PyTorch uses cross-correlation
        CUDNN_DATA_FLOAT
    ));

    // Determine algorithm for backward data (transposed conv)
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1; // default; can be tuned
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        filter_desc,
        input_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size
    ));

    // Allocate workspace if needed
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        void* raw_ws = nullptr;
        if (cudaMalloc(&raw_ws, workspace_size) != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed for workspace");
        }
        workspace_ptr = raw_ws;
    }

    // Perform backward data convolution: dy (input) * filter -> dx (output)
    // Here dy = input, filter = weight, dx = output (before bias)
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        filter_desc,
        weight_contig.data_ptr<float>(),
        input_desc,
        input_contig.data_ptr<float>(),
        conv_desc,
        algo,
        workspace_ptr,
        workspace_size,
        &beta,
        output_desc,
        output.data_ptr<float>()
    ));

    // Free workspace
    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }

    // Add bias if present (broadcast over N and L)
    if (bias_contig.defined()) {
        // We can use a simple kernel or cudaMemcpy2D; here we use a small raw CUDA kernel in .cu? But instructions:
        // .cpp cannot contain CUDA kernels. We could use a separate launcher, but for simplicity we can use
        // cudaMemset and a simple loop? Actually we cannot write CUDA kernel in .cpp. However, we can use cudaMemcpy2D
        // to copy bias to each row? That is inefficient. Better: we can call a helper from .cu that adds bias.
        // Since we must not add new .cu, we can implement bias addition using a torch operation? Not allowed.
        // Instead, we can set beta = 1 and pre-fill output with bias? But bias must be added after conv.
        // We can use cudaMemcpy2D to initialize output with bias repeated, but output is [N,C_out,1,L_out] and bias is [C_out].
        // That would require copying bias to each (n, l) position. We can use cudaMemcpy2D with pitch and src pointer offset.
        // However, this is cumbersome. Alternative: we can rely on the fact that the conv output is zero before beta=0; we can
        // set beta=1 and pre-fill output with bias broadcast using a torch operation? Not allowed.
        // Simpler: we can call a CUDA kernel from .cu that adds bias; we need to declare it. Since we cannot modify .cu now,
        // we need to assume such a kernel exists. But the current .cu only has softmax launcher.
        // Given the constraints, we can use cudaMemset and a series of cudaMemcpy2D operations to initialize output with bias
        // before calling cudnnConvolutionBackwardData with beta=1? But cudnn will overwrite with dx + beta*out.
        // Actually we can use cudaMemcpy2D to copy bias into each row of output? That would be extremely inefficient.
        // A better approach: after conv, we can use a simple loop over C_out and use cudaMemcpy to copy bias value to all N*L elements? No.
        // Maybe we can use cudaMemset to zero output, then use a custom kernel. But we can't add custom kernel to .cpp.
        // However, we CAN add a small launcher to .cu later, but currently .cu does not have it. The instructions say this phase only .cpp,
        // but we can declare an extern "C" function that we expect to exist in .cu. But it doesn't exist yet, so linking will fail.
        // The only way to add bias without a kernel is to use cublas? cublas can do rank-1 update? Not directly.
        // We could set beta=1 and initialize output with bias using cudaMemcpy2DAsync for each channel? That would be N*L copies per channel? No.
        // Actually we can use cudaMemset to set output to bias? But bias is different per channel.
        // We could use a grid of lambda functions? Not possible in C++.
        // So the clean solution is to assume that we will add a bias addition kernel to custom_ops.cu in a later phase, but for now declare it.
        // However, the task says this phase should produce a compilable binding. Since we cannot modify .cu, we must find a way without a custom kernel.
        // One trick: use cudaMemset to zero output, then use cudnnConvolutionBackwardData with beta=0 (already done), then use cudaMemcpy2D to add bias:
        // For each channel c, we can copy bias[c] into a temporary scalar and then use cudaMemcpy2D to copy that scalar to all positions? That would be many copies.
        // Alternatively, use cudaMemcpyToSymbol? Not helpful.
        // Maybe cudnnAddTensor can add bias to each channel? cudnnAddTensor can broadcast a tensor to another tensor with same number of elements,
        // but broadcasting over N and L requires creating a tensor with shape [1,C_out,1,1] and using cudnnAddTensor with alpha/beta, it will broadcast.
        // cudnnAddTensor supports broadcasting over dimensions with size 1. We can create a tensor descriptor for bias [1, C_out, 1, 1] and call cudnnAddTensor.
        // That is a clean solution! We'll use cudnnAddTensor.
        if (bias_contig.defined()) {
            cudnnTensorDescriptor_t bias_desc;
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
            // bias shape for broadcasting: [1, C_out, 1, 1]
            CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1));
            float alpha_bias = 1.0f;
            float beta_bias = 1.0f;
            CUDNN_CHECK(cudnnAddTensor(
                handle,
                &alpha_bias,
                bias_desc,
                bias_contig.data_ptr<float>(),
                &beta_bias,
                output_desc,
                output.data_ptr<float>()
            ));
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
        }
    }

    // Cleanup descriptors
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));

    // Note: output shape [N, C_out, L_out] and contiguous; strides will be [C_out*L_out, L_out, 1]
    return output;
}

// Softmax along dim 1 wrapper
// input: [N, C, L] any strides; output: new contiguous tensor
torch::Tensor softmax_dim1_ext(const torch::Tensor& input) {
    TORCH_CHECK(input.is_cuda(), "softmax_dim1_ext: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "softmax_dim1_ext: input must be float32");
    TORCH_CHECK(input.dim() == 3, "softmax_dim1_ext: input must be 3D (N, C, L)");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);

    // Allocate output contiguous
    auto output = torch::empty_like(input);
    // Note: empty_like preserves memory format? It will produce same strides as input if it's non-overlapping and dense? Actually empty_like gives contiguous or same as input? It gives same strides as input if memory format is preserve. But we need output to be contiguous? Actually softmax output should be contiguous per contract? The reference output shape is contiguous (strides [262144, 16384, 1]). So we should allocate contiguous output.
    // Better: use torch::empty({N,C,L}, input.options()) to guarantee contiguous.
    output = torch::empty({N, C, L}, input.options());

    // Get strides: for input, we need actual strides; for output, it's contiguous strides {C*L, L, 1}.
    long long in_stride_n = input.stride(0);
    long long in_stride_c = input.stride(1);
    long long in_stride_l = input.stride(2);
    long long out_stride_n = output.stride(0); // should be C*L
    long long out_stride_c = output.stride(1); // L
    long long out_stride_l = output.stride(2); // 1

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_dim1_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)N, (int)C, (int)L,
        in_stride_n, in_stride_c, in_stride_l,
        out_stride_n, out_stride_c, out_stride_l,
        stream
    );

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_ext", &conv_transpose1d_ext,
          "ConvTranspose1d using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"));
    m.def("softmax_dim1_ext", &softmax_dim1_ext,
          "Softmax along dim 1",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
