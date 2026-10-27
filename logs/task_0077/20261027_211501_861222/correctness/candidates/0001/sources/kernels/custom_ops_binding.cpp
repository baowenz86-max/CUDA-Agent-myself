#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
void layer_norm_forward_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    int N,
    int C,
    int L,
    float eps,
    cudaStream_t stream
);

namespace {

void check_cudnn(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

void check_cuda(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// RAII wrapper for cuDNN handle
class CudnnHandle {
public:
    CudnnHandle() {
        check_cudnn(cudnnCreate(&handle), "cudnnCreate failed");
    }
    ~CudnnHandle() {
        cudnnDestroy(handle);
    }
    cudnnHandle_t get() const { return handle; }
private:
    cudnnHandle_t handle;
};

// Global handle (created once, reused; safe because we set stream each call)
CudnnHandle& get_cudnn_handle() {
    static CudnnHandle handle;
    return handle;
}

} // anonymous namespace

// Python wrapper for conv_transpose1d_forward using cuDNN
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t padding,
    int64_t output_padding,
    int64_t groups,
    int64_t dilation
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must have 3 dimensions [N, C, L]");
    TORCH_CHECK(weight.dim() == 3, "weight must have 3 dimensions [in_channels, out_channels/groups, kernel_size]");
    TORCH_CHECK(bias.dim() == 1, "bias must have 1 dimension [out_channels]");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(groups > 0, "groups must be positive");
    TORCH_CHECK(dilation > 0, "dilation must be positive");

    // Make contiguous if necessary (cuDNN requires contiguous or specific strides)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    int64_t N = input_contig.size(0);
    int64_t C_in = input_contig.size(1);
    int64_t L_in = input_contig.size(2);
    int64_t C_out = weight_contig.size(1) * groups;
    int64_t kernel_size = weight_contig.size(2);

    TORCH_CHECK(weight_contig.size(0) == C_in, "weight shape mismatch: expected [", C_in, ", ", C_out / groups, ", ", kernel_size, "]");
    TORCH_CHECK(bias_contig.numel() == C_out, "bias size mismatch: expected ", C_out);

    int64_t L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;
    TORCH_CHECK(L_out > 0, "computed output length is non-positive");

    auto output = torch::empty({N, C_out, L_out}, input_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CudnnHandle& handle = get_cudnn_handle();
    check_cudnn(cudnnSetStream(handle.get(), stream), "cudnnSetStream failed");

    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filt_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&in_desc), "cudnnCreateTensorDescriptor failed");
    check_cudnn(cudnnCreateTensorDescriptor(&out_desc), "cudnnCreateTensorDescriptor failed");
    check_cudnn(cudnnCreateFilterDescriptor(&filt_desc), "cudnnCreateFilterDescriptor failed");
    check_cudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor failed");

    // Set tensor descriptors: cuDNN uses NCHW layout, so for 1D we use N, C, L, and dummy height 1
    check_cudnn(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)N, (int)C_in, (int)L_in, 1), "cudnnSetTensor4dDescriptor in failed");
    check_cudnn(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           (int)N, (int)C_out, (int)L_out, 1), "cudnnSetTensor4dDescriptor out failed");
    // cuDNN filter shape: [K, C, H, W] = [C_out, C_in/groups, L, 1]
    check_cudnn(cudnnSetFilter4dDescriptor(filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           (int)C_out, (int)(C_in / groups), (int)kernel_size, 1), "cudnnSetFilter4dDescriptor failed");

    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                0, 0, // pad_h, pad_w (0, because we handle padding manually in the invert transform? Actually we pass padding)
                                                (int)stride, (int)stride,
                                                (int)dilation, (int)dilation,
                                                CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor failed");

    // For backward data (transposed convolution), we need to set the convolution descriptor accordingly
    // Actually use cudnnSetConvolution2dDescriptor with mode CUDNN_CROSS_CORRELATION? Let's use default.
    // The transposed conv can be computed as conv backward data: we need to set pad = dilation*(kernel-1)-padding etc.
    // cuDNN has a special API for backward data that corresponds to transposed convolution.
    // We'll use cudnnConvolutionBackwardData.
    // Set the convolution descriptor as forward conv: pad = padding, stride = stride, dilation = dilation.
    check_cudnn(cudnnSetConvolution2dDescriptor(conv_desc,
                                                (int)padding, (int)padding,
                                                (int)stride, (int)stride,
                                                (int)dilation, (int)dilation,
                                                CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT), "cudnnSetConvolution2dDescriptor failed");

    // Algorithms
    cudnnConvolutionBwdDataAlgo_t algo;
    check_cudnn(cudnnGetConvolutionBackwardDataAlgorithm(handle.get(),
                                                         filt_desc,
                                                         in_desc,
                                                         conv_desc,
                                                         out_desc,
                                                         CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                                         0, &algo), "cudnnGetConvolutionBackwardDataAlgorithm failed");

    size_t workspace_size = 0;
    check_cudnn(cudnnGetConvolutionBackwardDataWorkspaceSize(handle.get(),
                                                             filt_desc,
                                                             in_desc,
                                                             conv_desc,
                                                             out_desc,
                                                             algo,
                                                             &workspace_size), "cudnnGetConvolutionBackwardDataWorkspaceSize failed");

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        check_cuda(cudaMalloc(&workspace, workspace_size), "cudaMalloc workspace failed");
    }

    float alpha = 1.0f, beta = 0.0f;
    check_cudnn(cudnnConvolutionBackwardData(handle.get(),
                                             &alpha,
                                             filt_desc, weight_contig.data_ptr<float>(),
                                             in_desc, input_contig.data_ptr<float>(),
                                             conv_desc, algo,
                                             workspace, workspace_size,
                                             &beta,
                                             out_desc, output.data_ptr<float>()), "cudnnConvolutionBackwardData failed");

    // Add bias: each output channel gets a bias value. We can use cudnnAddTensor or a simple kernel; but since bias is small, we can use cudnnAddTensor.
    // Create bias tensor descriptor (dims [1, C_out, 1, 1])
    cudnnTensorDescriptor_t bias_desc;
    check_cudnn(cudnnCreateTensorDescriptor(&bias_desc), "cudnnCreateTensorDescriptor bias failed");
    check_cudnn(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, (int)C_out, 1, 1), "cudnnSetTensor4dDescriptor bias failed");
    const float beta_bias = 1.0f; // add to existing output
    check_cudnn(cudnnAddTensor(handle.get(), &alpha, bias_desc, bias_contig.data_ptr<float>(), &beta_bias, out_desc, output.data_ptr<float>()), "cudnnAddTensor bias failed");

    // Cleanup
    if (workspace) {
        cudaFree(workspace);
    }
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    cudnnDestroyFilterDescriptor(filt_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(bias_desc);

    return output;
}

// Python wrapper for layer_norm_forward
torch::Tensor layer_norm_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 3, "input must have 3 dimensions [N, C, L]");
    TORCH_CHECK(weight.dim() == 2, "weight must have 2 dimensions [C, L]");
    TORCH_CHECK(bias.dim() == 2, "bias must have 2 dimensions [C, L]");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t L = input.size(2);
    TORCH_CHECK(weight.size(0) == C && weight.size(1) == L, "weight shape must match [C, L]");
    TORCH_CHECK(bias.size(0) == C && bias.size(1) == L, "bias shape must match [C, L]");

    // Make contiguous if necessary
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    auto output = torch::empty_like(input_contig);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    layer_norm_forward_launcher(
        output.data_ptr<float>(),
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        (int)N,
        (int)C,
        (int)L,
        (float)eps,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "layer_norm_forward_launcher failed: ", cudaGetErrorString(err));

    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("output_padding"),
          pybind11::arg("groups"),
          pybind11::arg("dilation"));

    m.def("layer_norm_forward", &layer_norm_forward,
          "LayerNorm forward over last two dimensions",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
