#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdint>

#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void softplus_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream);

// Helper to get current stream and cudnn handle
typedef struct CudnnContext {
    cudnnHandle_t handle;
    cudaStream_t stream;
    bool valid;
} CudnnContext;

static CudnnContext g_cudnn_ctx;
static bool g_cudnn_initialized = false;

static cudnnHandle_t get_cudnn_handle(cudaStream_t stream) {
    if (!g_cudnn_initialized) {
        cudnnCreate(&g_cudnn_ctx.handle);
        g_cudnn_ctx.valid = true;
        g_cudnn_initialized = true;
    }
    if (g_cudnn_ctx.stream != stream) {
        cudnnSetStream(g_cudnn_ctx.handle, stream);
        g_cudnn_ctx.stream = stream;
    }
    return g_cudnn_ctx.handle;
}

static void checkCudnn(cudnnStatus_t status, const char* msg) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(msg) + ": " + cudnnGetErrorString(status));
    }
}

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(err));
    }
}

// Convert PyTorch tensor to cudnnTensorDescriptor_t with appropriate format and strides
torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    std::vector<int64_t> stride,
    std::vector<int64_t> padding,
    std::vector<int64_t> dilation
) {
    // Check device and dtype
    TORCH_CHECK(input.is_cuda(), "conv3d_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv3d_forward: weight must be a CUDA tensor");
    if (bias.has_value() && bias.value().is_cuda()) {
        TORCH_CHECK(bias.value().is_cuda(), "conv3d_forward: bias must be a CUDA tensor");
    }
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv3d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv3d_forward: weight must be float32");
    if (bias.has_value()) {
        TORCH_CHECK(bias.value().dtype() == torch::kFloat32, "conv3d_forward: bias must be float32");
    }

    // Ensure input and weight are contiguous for cuDNN (can also support strides, but simplest to handle contiguous)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();

    const int dim = 5; // 3D convolution
    int64_t N = input_contig.size(0);
    int64_t C = input_contig.size(1);
    int64_t D = input_contig.size(2);
    int64_t H = input_contig.size(3);
    int64_t W = input_contig.size(4);

    int64_t K = weight_contig.size(0); // output channels
    int64_t C_in = weight_contig.size(1); // should equal C
    int64_t kD = weight_contig.size(2);
    int64_t kH = weight_contig.size(3);
    int64_t kW = weight_contig.size(4);

    TORCH_CHECK(C_in == C, "conv3d_forward: input channels mismatch");

    // stride, padding, dilation vectors must have 3 elements
    TORCH_CHECK(stride.size() == 3, "conv3d_forward: stride must have 3 elements");
    TORCH_CHECK(padding.size() == 3, "conv3d_forward: padding must have 3 elements");
    TORCH_CHECK(dilation.size() == 3, "conv3d_forward: dilation must have 3 elements");

    // Compute output dimensions
    auto calc_out = [](int64_t in_size, int64_t k, int64_t stride, int64_t pad, int64_t dilation) -> int64_t {
        return (in_size + 2 * pad - dilation * (k - 1) - 1) / stride + 1;
    };
    int64_t D_out = calc_out(D, kD, stride[0], padding[0], dilation[0]);
    int64_t H_out = calc_out(H, kH, stride[1], padding[1], dilation[1]);
    int64_t W_out = calc_out(W, kW, stride[2], padding[2], dilation[2]);

    auto output = torch::empty({N, K, D_out, H_out, W_out}, input_contig.options());

    // cuDNN descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    checkCudnn(cudnnCreateTensorDescriptor(&x_desc), "cudnnCreateTensorDescriptor x");
    checkCudnn(cudnnCreateTensorDescriptor(&y_desc), "cudnnCreateTensorDescriptor y");
    checkCudnn(cudnnCreateFilterDescriptor(&w_desc), "cudnnCreateFilterDescriptor");
    checkCudnn(cudnnCreateConvolutionDescriptor(&conv_desc), "cudnnCreateConvolutionDescriptor");

    // Set tensor descriptors (NCHW or NCDHW for 3D? cuDNN uses NCDHW for 5D)
    // For 3D convolution, cuDNN uses 5D tensors in NCDHW format.
    int dimA[5];
    int strideA[5];
    // input: N, C, D, H, W
    dimA[0] = static_cast<int>(N);
    dimA[1] = static_cast<int>(C);
    dimA[2] = static_cast<int>(D);
    dimA[3] = static_cast<int>(H);
    dimA[4] = static_cast<int>(W);
    strideA[0] = static_cast<int>(C * D * H * W);
    strideA[1] = static_cast<int>(D * H * W);
    strideA[2] = static_cast<int>(H * W);
    strideA[3] = static_cast<int>(W);
    strideA[4] = 1;
    checkCudnn(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, dimA, strideA), "cudnnSetTensorNdDescriptor x");

    // output: N, K, D_out, H_out, W_out
    int dimB[5];
    int strideB[5];
    dimB[0] = static_cast<int>(N);
    dimB[1] = static_cast<int>(K);
    dimB[2] = static_cast<int>(D_out);
    dimB[3] = static_cast<int>(H_out);
    dimB[4] = static_cast<int>(W_out);
    strideB[0] = static_cast<int>(K * D_out * H_out * W_out);
    strideB[1] = static_cast<int>(D_out * H_out * W_out);
    strideB[2] = static_cast<int>(H_out * W_out);
    strideB[3] = static_cast<int>(W_out);
    strideB[4] = 1;
    checkCudnn(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, dimB, strideB), "cudnnSetTensorNdDescriptor y");

    // filter: K, C, kD, kH, kW
    int filterDimA[5];
    filterDimA[0] = static_cast<int>(K);
    filterDimA[1] = static_cast<int>(C);
    filterDimA[2] = static_cast<int>(kD);
    filterDimA[3] = static_cast<int>(kH);
    filterDimA[4] = static_cast<int>(kW);
    checkCudnn(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, filterDimA), "cudnnSetFilterNdDescriptor");

    // convolution descriptor
    int padA[3] = {static_cast<int>(padding[0]), static_cast<int>(padding[1]), static_cast<int>(padding[2])};
    int filterStrideA[3] = {static_cast<int>(stride[0]), static_cast<int>(stride[1]), static_cast<int>(stride[2])};
    int dilationA[3] = {static_cast<int>(dilation[0]), static_cast<int>(dilation[1]), static_cast<int>(dilation[2])};
    checkCudnn(cudnnSetConvolutionNdDescriptor(conv_desc, 3, padA, filterStrideA, dilationA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "cudnnSetConvolutionNdDescriptor");

    // Get algorithm
    cudnnConvolutionFwdAlgo_t algo;
    checkCudnn(cudnnGetConvolutionForwardAlgorithm(get_cudnn_handle(c10::cuda::getCurrentCUDAStream().stream()),
                                                   x_desc, w_desc, conv_desc, y_desc,
                                                   CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo), "cudnnGetConvolutionForwardAlgorithm");

    // Workspace
    size_t workspace_size = 0;
    checkCudnn(cudnnGetConvolutionForwardWorkspaceSize(get_cudnn_handle(c10::cuda::getCurrentCUDAStream().stream()),
                                                        x_desc, w_desc, conv_desc, y_desc, algo, &workspace_size), "cudnnGetConvolutionForwardWorkspaceSize");
    void* workspace = nullptr;
    if (workspace_size > 0) {
        workspace = torch::empty({static_cast<int64_t>(workspace_size)}, input_contig.options().dtype(torch::kUInt8)).data_ptr();
    }

    float alpha = 1.0f, beta = 0.0f;
    checkCudnn(cudnnConvolutionForward(get_cudnn_handle(c10::cuda::getCurrentCUDAStream().stream()),
                                        &alpha,
                                        x_desc, input_contig.data_ptr<float>(),
                                        w_desc, weight_contig.data_ptr<float>(),
                                        conv_desc, algo,
                                        workspace, workspace_size,
                                        &beta,
                                        y_desc, output.data_ptr<float>()), "cudnnConvolutionForward");

    // Add bias if present
    if (bias.has_value() && bias.value().defined()) {
        auto bias_contig = bias.value().contiguous();
        TORCH_CHECK(bias_contig.size(0) == K, "conv3d_forward: bias size mismatch");
        // For bias addition, use cudnnAddTensor
        if (b_desc) {
            cudnnDestroyTensorDescriptor(b_desc);
        }
        checkCudnn(cudnnCreateTensorDescriptor(&b_desc), "cudnnCreateTensorDescriptor bias");
        int biasDim[5] = {1, static_cast<int>(K), 1, 1, 1};
        int biasStride[5] = {static_cast<int>(K), 1, 1, 1, 1};
        checkCudnn(cudnnSetTensorNdDescriptor(b_desc, CUDNN_DATA_FLOAT, 5, biasDim, biasStride), "cudnnSetTensorNdDescriptor bias");
        float alpha_bias = 1.0f;
        checkCudnn(cudnnAddTensor(get_cudnn_handle(c10::cuda::getCurrentCUDAStream().stream()),
                                  &alpha_bias, b_desc, bias_contig.data_ptr<float>(),
                                  &alpha, y_desc, output.data_ptr<float>()), "cudnnAddTensor");
    }

    // Cleanup descriptors
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    if (bias.has_value() && bias.value().defined()) {
        cudnnDestroyTensorDescriptor(b_desc);
    }
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return output;
}

torch::Tensor softplus_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "softplus_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "softplus_forward: input must be float32");
    // softplus is elementwise, so non-contiguous is fine; kernel assumes contiguous, so make contiguous
    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);
    int64_t numel = input_contig.numel();
    if (numel == 0) return output;
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softplus_launcher(input_contig.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    // kernel launch error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("softplus_launcher failed: ") + cudaGetErrorString(err));
    }
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("conv3d_forward", &conv3d_forward,
          "3D convolution forward (cuDNN)",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"),
          pybind11::arg("dilation"));
    m.def("softplus_forward", &softplus_forward,
          "Softplus activation forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
