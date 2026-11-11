// kernels/custom_ops_binding.cpp

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <cstdint>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void gelu_forward_launcher(
    const float* input,
    float* output,
    int64_t numel,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3, int64_t dim4,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
    cudaStream_t stream);

extern "C" void bias_add_forward_launcher(
    const float* input,
    const float* bias,
    float* output,
    int64_t numel,
    int64_t C,
    int64_t D, int64_t H, int64_t W, int64_t N,
    int64_t in_stride0, int64_t in_stride1, int64_t in_stride2, int64_t in_stride3, int64_t in_stride4,
    int64_t out_stride0, int64_t out_stride1, int64_t out_stride2, int64_t out_stride3, int64_t out_stride4,
    cudaStream_t stream);

// Helper function to convert Python int/tuple to int array of size 3
static std::vector<int> parse_3_int(const pybind11::handle& obj, const char* name) {
    std::vector<int> result(3);
    if (pybind11::isinstance<pybind11::int_>(obj)) {
        int v = obj.cast<int>();
        result = {v, v, v};
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        auto tup = obj.cast<pybind11::tuple>();
        TORCH_CHECK(tup.size() == 3, name, " must be int or tuple of 3 ints");
        for (int i = 0; i < 3; ++i) {
            result[i] = tup[i].cast<int>();
        }
    } else {
        TORCH_CHECK(false, name, " must be int or tuple of 3 ints");
    }
    return result;
}

// cuDNN error checking macro
#define CUDNN_CHECK(expr)                                                        \
    do {                                                                          \
        cudnnStatus_t status = (expr);                                           \
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS,                              \
                    "cuDNN error at ", __FILE__, ":", __LINE__, ": ",          \
                    cudnnGetErrorString(status));                                \
    } while (0)

// conv3d_forward wrapper
static torch::Tensor conv3d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    pybind11::object stride,
    pybind11::object padding) {

    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(weight.dim() == 5, "weight must be 5D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    TORCH_CHECK(weight.size(1) == input.size(1), "input/weight channel mismatch");
    TORCH_CHECK(weight.size(0) == bias.size(0), "weight/bias channel mismatch");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    // Parse stride and padding
    std::vector<int> stride_arr = parse_3_int(stride, "stride");
    std::vector<int> padding_arr = parse_3_int(padding, "padding");
    const int strideD = stride_arr[0], strideH = stride_arr[1], strideW = stride_arr[2];
    const int padD = padding_arr[0], padH = padding_arr[1], padW = padding_arr[2];

    // Get dimensions
    const int N = input.size(0);
    const int C_in = input.size(1);
    const int D = input.size(2);
    const int H = input.size(3);
    const int W = input.size(4);
    const int C_out = weight.size(0);
    const int kD = weight.size(2);
    const int kH = weight.size(3);
    const int kW = weight.size(4);

    const int D_out = (D + 2 * padD - kD) / strideD + 1;
    const int H_out = (H + 2 * padH - kH) / strideH + 1;
    const int W_out = (W + 2 * padW - kW) / strideW + 1;

    // Allocate output (contiguous)
    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle (per call but fine)
    cudnnHandle_t cudnn_handle;
    CUDNN_CHECK(cudnnCreate(&cudnn_handle));
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, stream));

    // Tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));

    // Filter descriptor
    cudnnFilterDescriptor_t w_desc;
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&w_desc));

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors: NCDHW is standard
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(x_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N,
                                           C_in,
                                           D,
                                           H * W));
    // Need 5D? Actually cuDNN requires dims differently, use SetTensorNdDescriptor.
    // For clarity we use the 5D version with CUDNN_TENSOR_NCDHW.
    // Recreate properly:
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);

    // Recreate with 5D
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&b_desc));

    int x_dims[5] = {N, C_in, D, H, W};
    int x_strides[5] = {C_in*D*H*W, D*H*W, H*W, W, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 5, x_dims, x_strides));

    int y_dims[5] = {N, C_out, D_out, H_out, W_out};
    int y_strides[5] = {C_out*D_out*H_out*W_out, D_out*H_out*W_out, H_out*W_out, W_out, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 5, y_dims, y_strides));

    int b_dims[5] = {1, C_out, 1, 1, 1};
    int b_strides[5] = {C_out, 1, 1, 1, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(b_desc, CUDNN_DATA_FLOAT, 5, b_dims, b_strides));

    // Filter descriptor: weight shape [C_out, C_in, kD, kH, kW]
    int w_dims[5] = {C_out, C_in, kD, kH, kW};
    CUDNN_CHECK(cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5, w_dims));

    // Convolution descriptor
    int conv_pad[3] = {padD, padH, padW};
    int conv_stride[3] = {strideD, strideH, strideW};
    int conv_dilation[3] = {1, 1, 1};
    CUDNN_CHECK(cudnnSetConvolutionNdDescriptor(conv_desc,
                                                3,
                                                conv_pad,
                                                conv_stride,
                                                conv_dilation,
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Algorithm selection
    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn_handle,
                                                     x_desc,
                                                     w_desc,
                                                     conv_desc,
                                                     y_desc,
                                                     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                     0,
                                                     &algo));

    // Workspace size
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                         x_desc,
                                                         w_desc,
                                                         conv_desc,
                                                         y_desc,
                                                         algo,
                                                         &workspace_size));

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(workspace != nullptr, "cudaMalloc failed for workspace");
    }

    // Perform convolution
    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn_handle,
                                        &alpha,
                                        x_desc,
                                        input.data_ptr<float>(),
                                        w_desc,
                                        weight.data_ptr<float>(),
                                        conv_desc,
                                        algo,
                                        workspace,
                                        workspace_size,
                                        &beta,
                                        y_desc,
                                        output.data_ptr<float>()));

    // Add bias (beta = 1.0)
    beta = 1.0f;
    CUDNN_CHECK(cudnnAddTensor(cudnn_handle,
                               &alpha,
                               b_desc,
                               bias.data_ptr<float>(),
                               &beta,
                               y_desc,
                               output.data_ptr<float>()));

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// gelu_forward wrapper
static torch::Tensor gelu_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");

    auto output = torch::empty_like(input);

    int64_t numel = input.numel();
    if (numel == 0) return output;

    // Get dimensions and strides
    std::vector<int64_t> dims, input_strides, output_strides;
    for (int i = 0; i < input.dim(); ++i) {
        dims.push_back(input.size(i));
        input_strides.push_back(input.stride(i));
        output_strides.push_back(output.stride(i));
    }
    // If tensor has fewer than 5 dims, pad leading dims with 1 and stride 0? Actually we assume 5D based on contract.
    if (input.dim() != 5) {
        TORCH_CHECK(false, "gelu_forward expects 5D tensor as per contract");
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    gelu_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        dims[0], dims[1], dims[2], dims[3], dims[4],
        input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
        stream
    );

    return output;
}

// bias_add_forward wrapper
static torch::Tensor bias_add_forward(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    // bias shape is [C] or [C,1,1,1] but effectively length C
    TORCH_CHECK(bias.numel() == input.size(1), "bias size must match channel dim");

    auto output = torch::empty_like(input);

    int64_t numel = input.numel();
    if (numel == 0) return output;

    const int64_t N = input.size(0);
    const int64_t C = input.size(1);
    const int64_t D = input.size(2);
    const int64_t H = input.size(3);
    const int64_t W = input.size(4);

    // Get strides
    std::vector<int64_t> input_strides(5), output_strides(5);
    for (int i = 0; i < 5; ++i) {
        input_strides[i] = input.stride(i);
        output_strides[i] = output.stride(i);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_forward_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output.data_ptr<float>(),
        numel,
        C,
        D, H, W, N,
        input_strides[0], input_strides[1], input_strides[2], input_strides[3], input_strides[4],
        output_strides[0], output_strides[1], output_strides[2], output_strides[3], output_strides[4],
        stream
    );

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("conv3d_forward", &conv3d_forward,
          "Conv3d forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("padding"));
    m.def("gelu_forward", &gelu_forward,
          "GELU activation forward",
          pybind11::arg("input"));
    m.def("bias_add_forward", &bias_add_forward,
          "Bias add forward",
          pybind11::arg("input"),
          pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
