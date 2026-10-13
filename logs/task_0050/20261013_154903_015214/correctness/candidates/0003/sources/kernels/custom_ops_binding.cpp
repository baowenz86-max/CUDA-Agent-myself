#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Helper to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t err = call; \
    TORCH_CHECK(err == CUDNN_STATUS_SUCCESS, "cuDNN error: ", cudnnGetErrorString(err)); \
}

// Temporary helper to create cuDNN tensor descriptor for NCHW float32 tensor with optional strides
static cudnnTensorDescriptor_t createTensorDesc(const torch::Tensor& tensor) {
    cudnnTensorDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    
    auto sizes = tensor.sizes();
    auto strides = tensor.strides();
    int n = sizes[0], c = sizes[1], h = sizes[2], w = sizes[3];
    
    std::vector<int> dims = {n, c, h, w};
    std::vector<int> strides_arr = {
        (int)strides[0], (int)strides[1], (int)strides[2], (int)strides[3]
    };
    
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        desc,
        CUDNN_DATA_FLOAT,
        4,
        dims.data(),
        strides_arr.data()
    ));
    return desc;
}

// Creates a filter descriptor for NCHW layout, using strides if provided
static cudnnFilterDescriptor_t createFilterDesc(const torch::Tensor& weight) {
    cudnnFilterDescriptor_t desc;
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc));
    
    auto sizes = weight.sizes();
    auto strides = weight.strides();
    int out_c = sizes[0], in_c_per_group = sizes[1], k_h = sizes[2], k_w = sizes[3];
    std::vector<int> dims = {out_c, in_c_per_group, k_h, k_w};
    
    CHECK_CUDNN(cudnnSetFilterNdDescriptor(
        desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        4,
        dims.data()
    ));
    return desc;
}

// Conv2d forward using cuDNN
// weight: [out_channels, in_channels/groups, kernel_h, kernel_w]
// input: [N, in_channels, H, W]
// Returns: [N, out_channels, H_out, W_out]
torch::Tensor conv2d_forward_cudnn(
    const torch::Tensor& weight,
    const torch::Tensor& input,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    int64_t groups
) {
    // Parameter validation
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N,C,H,W)");
    TORCH_CHECK(weight.dim() == 4, "weight must be 4D (out_c, in_c/groups, kH, kW)");
    TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride must be positive");
    TORCH_CHECK(padding_h >= 0 && padding_w >= 0, "padding must be non-negative");
    TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation must be positive");
    TORCH_CHECK(groups > 0, "groups must be positive");
    
    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);
    int64_t out_c = weight.size(0);
    int64_t in_c_per_group = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);
    
    TORCH_CHECK(C == in_c_per_group * groups, "input channels must equal in_c_per_group * groups");
    TORCH_CHECK(out_c % groups == 0, "out_channels must be divisible by groups");
    
    int64_t H_out = (H + 2 * padding_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * padding_w - dilation_w * (kW - 1) - 1) / stride_w + 1;
    TORCH_CHECK(H_out > 0 && W_out > 0, "output size must be positive");
    
    // Make input and weight contiguous if not already (cuDNN requires contiguous for these APIs)
    torch::Tensor input_contig = input.contiguous();
    torch::Tensor weight_contig = weight.contiguous();
    
    // Allocate output tensor of shape [N, out_c, H_out, W_out]
    auto output = torch::empty({N, out_c, H_out, W_out}, input.options());
    
    // Get cuDNN handle and stream
    cudnnHandle_t cudnn_handle;
    CHECK_CUDNN(cudnnCreate(&cudnn_handle));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    CHECK_CUDNN(cudnnSetStream(cudnn_handle, stream));
    
    // Set up descriptors
    cudnnTensorDescriptor_t input_desc = createTensorDesc(input_contig);
    cudnnTensorDescriptor_t output_desc = createTensorDesc(output);
    cudnnFilterDescriptor_t filter_desc = createFilterDesc(weight_contig);
    
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        conv_desc,
        padding_h, padding_w,
        stride_h, stride_w,
        dilation_h, dilation_w,
        CUDNN_CROSS_CORRELATION,  // PyTorch conv2d uses cross-correlation
        CUDNN_DATA_FLOAT
    ));
    if (groups > 1) {
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(conv_desc, groups));
    }
    
    // Determine algorithm (use deterministic algorithm, may be slower but correct)
    cudnnConvolutionFwdAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        0,
        &algo
    ));
    
    // Workspace size
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size
    ));
    
    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }
    
    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(
        cudnn_handle,
        &alpha,
        input_desc, input_contig.data_ptr<float>(),
        filter_desc, weight_contig.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        output_desc, output.data_ptr<float>()
    ));
    
    // Cleanup
    if (workspace) {
        CHECK_CUDA(cudaFree(workspace));
    }
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn_handle));
    
    return output;
}

// Registration
void register_conv2d_forward(pybind11::module& m) {
    m.def("conv2d_forward", &conv2d_forward_cudnn,
          "conv2d forward using cuDNN",
          py::arg("weight"),
          py::arg("input"),
          py::arg("stride_h"),
          py::arg("stride_w"),
          py::arg("padding_h"),
          py::arg("padding_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"),
          py::arg("groups"));
}

REGISTER_BINDING(conv2d_forward, register_conv2d_forward);
