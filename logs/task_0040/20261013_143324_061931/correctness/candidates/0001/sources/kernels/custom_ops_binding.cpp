// Binding for custom ops defined in custom_ops.cu and cuDNN conv_transpose1d.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void channel_add_launcher(const float* x, const float* bias, float* y,
                                     int64_t N, int64_t C, int64_t L, cudaStream_t stream);
extern "C" void roll_last_dim_launcher(const float* x, float* y,
                                       int64_t N, int64_t C, int64_t L, int64_t shift, cudaStream_t stream);
extern "C" void linear_forward_launcher(const float* x, const float* weight, const float* bias,
                                        float* y, int64_t N, int64_t in_features, int64_t out_features,
                                        cudaStream_t stream);
extern "C" void flatten_to_2d_launcher(cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentCUDAStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Check CUDA, dtype float32, and make contiguous if needed
torch::Tensor ensure_cuda_float_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    if (!t.is_contiguous()) {
        return t.contiguous();
    }
    return t;
}

// ===================== conv_transpose1d_forward =====================
torch::Tensor conv_transpose1d_forward(torch::Tensor x,
                                       torch::Tensor weight,
                                       torch::Tensor bias,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t output_padding,
                                       int64_t groups = 1,
                                       int64_t dilation = 1) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [N,C,L]");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D [C_in,C_out,kernel_size]");

    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias.dim() == 1, "bias must be 1D");
    }

    int64_t N = x.size(0);
    int64_t C_in = x.size(1);
    int64_t L_in = x.size(2);
    int64_t C_out = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == C_in, "weight shape mismatch");
    TORCH_CHECK(groups == 1, "only groups=1 supported");
    TORCH_CHECK(dilation == 1, "only dilation=1 supported");

    int64_t L_out = (L_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1;

    // Ensure contiguous tensors for cuDNN
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();
    torch::Tensor bias_contig;
    if (bias.defined()) {
        bias_contig = bias.contiguous();
    }

    auto output = torch::empty({N, C_out, L_out}, x.options());

    cudaStream_t stream = getCurrentCUDAStream();

    // cuDNN handle
    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);
    cudnnSetStream(cudnn_handle, stream);

    // Tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set tensor descriptors: NCHW format, but for 1D we treat as N,C,L,1
    // For 1D convolution, cuDNN requires 4D tensors with H=1, W=L. We'll use NCHW with H=1.
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_in, 1, L_in);
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C_out, 1, L_out);
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C_out, 1, 1);

    // Filter descriptor: weight shape [C_in, C_out, kernel_size]. Need to treat as [C_out, C_in/groups, kernel_size] for cuDNN.
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, C_out, C_in, 1, kernel_size);

    // Convolution descriptor: use cross-correlation with transpose
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    padding, padding,       // padH, padW (both padding)
                                    stride, stride,         // strideH, strideW
                                    dilation, dilation,     // dilationH, dilationW
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // For transposed convolution, we use cudnnConvolutionBackwardData
    cudnnSetConvolutionMathType(conv_desc, CUDNN_DEFAULT_MATH);

    float alpha = 1.0f, beta = 0.0f;

    // Compute gradients wrt data (transposed conv)
    cudnnConvolutionBackwardData(cudnn_handle,
                                 &alpha,
                                 w_desc, weight_contig.data_ptr<float>(),
                                 x_desc, x_contig.data_ptr<float>(),
                                 conv_desc,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, // choose algorithm; can be 0 for auto
                                 nullptr, 0,
                                 &beta,
                                 y_desc, output.data_ptr<float>());

    // Add bias if present
    if (bias_contig.defined()) {
        // Use cudnnAddTensor for channel-wise bias addition
        cudnnAddTensor(cudnn_handle, &alpha, b_desc, bias_contig.data_ptr<float>(),
                       &alpha, y_desc, output.data_ptr<float>());
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);

    return output;
}

// ===================== channel_add =====================
torch::Tensor channel_add(torch::Tensor x, torch::Tensor bias) {
    auto x_contig = ensure_cuda_float_contiguous(x, "x");
    auto bias_contig = ensure_cuda_float_contiguous(bias, "bias");

    TORCH_CHECK(x_contig.dim() == 3, "x must be 3D [N,C,L]");
    TORCH_CHECK(bias_contig.dim() == 2 && bias_contig.size(1) == 1, "bias must be [C,1]");
    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t L = x_contig.size(2);
    TORCH_CHECK(bias_contig.size(0) == C, "bias first dim must match channels");

    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = getCurrentCUDAStream();
    channel_add_launcher(x_contig.data_ptr<float>(),
                         bias_contig.data_ptr<float>(),
                         output.data_ptr<float>(),
                         N, C, L, stream);
    return output;
}

// ===================== roll_last_dim =====================
torch::Tensor roll_last_dim(torch::Tensor x, int64_t shift) {
    auto x_contig = ensure_cuda_float_contiguous(x, "x");
    TORCH_CHECK(x_contig.dim() == 3, "x must be 3D [N,C,L]");
    int64_t N = x_contig.size(0);
    int64_t C = x_contig.size(1);
    int64_t L = x_contig.size(2);

    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = getCurrentCUDAStream();
    roll_last_dim_launcher(x_contig.data_ptr<float>(),
                           output.data_ptr<float>(),
                           N, C, L, shift, stream);
    return output;
}

// ===================== flatten_to_2d =====================
torch::Tensor flatten_to_2d(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D [N,C,L]");
    // Return view: no data copy; x is contiguous from previous ops typically,
    // but if not, view will fail if non-contiguous. We can call contiguous if needed.
    if (!x.is_contiguous()) {
        x = x.contiguous();
    }
    return x.view({x.size(0), -1});
}

// ===================== linear_forward =====================
torch::Tensor linear_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias) {
    auto x_contig = ensure_cuda_float_contiguous(x, "x");
    auto weight_contig = ensure_cuda_float_contiguous(weight, "weight");
    TORCH_CHECK(x_contig.dim() == 2, "x must be 2D [N, in_features]");
    TORCH_CHECK(weight_contig.dim() == 2, "weight must be 2D [out_features, in_features]");
    int64_t N = x_contig.size(0);
    int64_t in_features = x_contig.size(1);
    int64_t out_features = weight_contig.size(0);
    TORCH_CHECK(weight_contig.size(1) == in_features, "weight shape mismatch");

    torch::Tensor bias_contig;
    if (bias.defined()) {
        bias_contig = ensure_cuda_float_contiguous(bias, "bias");
        TORCH_CHECK(bias_contig.dim() == 1 && bias_contig.size(0) == out_features, "bias shape mismatch");
    }

    auto output = torch::empty({N, out_features}, x_contig.options());
    cudaStream_t stream = getCurrentCUDAStream();
    linear_forward_launcher(x_contig.data_ptr<float>(),
                            weight_contig.data_ptr<float>(),
                            bias_contig.defined() ? bias_contig.data_ptr<float>() : nullptr,
                            output.data_ptr<float>(),
                            N, in_features, out_features, stream);
    return output;
}

// Register all ops
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"),
          py::arg("groups") = 1, py::arg("dilation") = 1);
    m.def("channel_add", &channel_add, "Add channel-wise bias",
          py::arg("x"), py::arg("bias"));
    m.def("roll_last_dim", &roll_last_dim, "Roll last dim by shift",
          py::arg("x"), py::arg("shift"));
    m.def("flatten_to_2d", &flatten_to_2d, "Flatten to 2D view",
          py::arg("x"));
    m.def("linear_forward", &linear_forward, "Linear forward",
          py::arg("x"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
