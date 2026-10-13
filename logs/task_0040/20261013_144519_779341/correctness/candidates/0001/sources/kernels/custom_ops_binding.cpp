#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// -----------------------------------------------------------------------------
// Declarations of launchers from custom_ops.cu
// -----------------------------------------------------------------------------
extern "C" void add_bias_launcher(float* output,
                                  const float* input,
                                  const float* bias,
                                  int channels,
                                  int length,
                                  long long total_elements,
                                  cudaStream_t stream);

extern "C" void roll_launcher(float* output,
                              const float* input,
                              int length,
                              int shifts,
                              long long total_elements,
                              cudaStream_t stream);

extern "C" void view_launcher(float* output,
                              const float* input,
                              long long total_elements,
                              cudaStream_t stream);

extern "C" void linear_launcher(float* output,
                                const float* input,
                                const float* weight,
                                const float* bias,
                                int batch_size,
                                int num_features,
                                int out_features,
                                cudaStream_t stream);

// -----------------------------------------------------------------------------
// Helper: check CUDA tensor and dtype
// -----------------------------------------------------------------------------
static void check_cuda_float(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// -----------------------------------------------------------------------------
// 1. ConvTranspose1d forward using cuDNN
// -----------------------------------------------------------------------------
torch::Tensor conv_transpose1d_forward(torch::Tensor input,
                                       torch::Tensor weight,
                                       torch::Tensor bias,
                                       int64_t stride,
                                       int64_t padding,
                                       int64_t output_padding) {
    check_cuda_float(input, "input");
    check_cuda_float(weight, "weight");
    check_cuda_float(bias, "bias");

    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D (out_channels)");

    int batch_size = input.size(0);
    int in_channels = input.size(1);
    int input_length = input.size(2);
    int out_channels = weight.size(1);
    int kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(bias.size(0) == out_channels, "bias out_channels mismatch");

    int output_length = (input_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(output_length > 0, "computed output length must be positive");

    // cuDNN requires contiguous tensors
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreate failed");

    // Set tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    status = cudnnCreateTensorDescriptor(&input_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor input failed");
    status = cudnnCreateTensorDescriptor(&output_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor output failed");
    status = cudnnCreateTensorDescriptor(&bias_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateTensorDescriptor bias failed");
    status = cudnnCreateFilterDescriptor(&filter_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateFilterDescriptor failed");
    status = cudnnCreateConvolutionDescriptor(&conv_desc);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnCreateConvolutionDescriptor failed");

    // For 1D convolution, cuDNN uses 2D descriptors with height=1
    status = cudnnSetTensor4dDescriptor(input_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        in_channels,
                                        1,
                                        input_length);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor input failed");

    status = cudnnSetTensor4dDescriptor(output_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        batch_size,
                                        out_channels,
                                        1,
                                        output_length);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor output failed");

    status = cudnnSetTensor4dDescriptor(bias_desc,
                                        CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT,
                                        1,
                                        out_channels,
                                        1,
                                        1);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetTensor4dDescriptor bias failed");

    // Filter descriptor: cuDNN expects [out_channels, in_channels/groups, kernel_h, kernel_w]
    // For ConvTranspose1d, weight shape is [in_channels, out_channels, kernel_size], so we need to permute.
    // Actually cuDNN expects filter as [K, C, H, W] where K = out_channels, C = in_channels/groups.
    // Our weight is [in_channels, out_channels, kernel_size], so we need to permute to [out_channels, in_channels, kernel_size, 1].
    // We'll create a temporary contiguous tensor in cuDNN layout.
    auto weight_cudnn = weight_contig.permute({1, 0, 2}).unsqueeze(3).contiguous();

    status = cudnnSetFilter4dDescriptor(filter_desc,
                                        CUDNN_DATA_FLOAT,
                                        CUDNN_TENSOR_NCHW,
                                        out_channels,
                                        in_channels,
                                        kernel_size,
                                        1);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetFilter4dDescriptor failed");

    // Convolution descriptor
    status = cudnnSetConvolution2dDescriptor(conv_desc,
                                             0, 0,  // pad_h, pad_w (we'll set manually? Actually use padding)
                                             stride, stride,
                                             1, 1,  // dilation
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    // For deconvolution, we need to set padding and output_padding separately.
    // cuDNN has cudnnSetConvolution2dDescriptor, but for transpose we may need to adjust.
    // Actually cudnnConvolutionBackwardData can be used for ConvTranspose.
    // We'll use cudnnConvolutionBackwardData.
    // So we need to set descriptors for the "gradient" operation.
    // Let's create a convolution descriptor for the forward convolution of the transposed operation.
    // We'll set the convolution descriptor as if doing a forward conv with the transposed weights.
    // Better approach: use cudnnConvolutionBackwardData with appropriate descriptors.

    // For cudnnConvolutionBackwardData:
    // dx = cudnnConvolutionBackwardData(handle, alpha, w_desc, w, dy_desc, dy, conv_desc, beta, dx_desc, dx)
    // Here dx corresponds to our input, dy to our output, w to our weight (transposed layout).
    // So we need to set conv_desc with same padding, stride as forward conv that would produce our output from input.
    // The forward conv would have input = our input, output = our output, stride = stride, padding = padding.
    // So conv_desc should be set with these parameters.
    // We need to set conv_desc to have pad_h = padding, pad_w = padding, stride_h = stride, stride_w = stride, dilation = 1.
    
    status = cudnnSetConvolution2dDescriptor(conv_desc,
                                             padding, padding,  // pad_h, pad_w
                                             stride, stride,
                                             1, 1,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnSetConvolution2dDescriptor failed");

    // Algorithm selection
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle,
                                             filter_desc,
                                             output_desc,
                                             conv_desc,
                                             input_desc,
                                             CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
                                             0,
                                             &algo);

    // Workspace size
    size_t workspace_size = 0;
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle,
                                                          filter_desc,
                                                          output_desc,
                                                          conv_desc,
                                                          input_desc,
                                                          algo,
                                                          &workspace_size);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnGetConvolutionBackwardDataWorkspaceSize failed");

    // Allocate workspace
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_err = cudaMalloc(&workspace, workspace_size);
        TORCH_CHECK(cuda_err == cudaSuccess, "cudaMalloc workspace failed");
    }

    // Create output tensor
    auto output = torch::empty({batch_size, out_channels, output_length}, input.options());

    // Get current stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(cudnn_handle, stream);

    float alpha = 1.0f, beta = 0.0f;
    status = cudnnConvolutionBackwardData(cudnn_handle,
                                          &alpha,
                                          filter_desc,
                                          weight_cudnn.data_ptr<float>(),
                                          output_desc,
                                          output.data_ptr<float>(),  // dy is our output (actually input to backward)
                                          conv_desc,
                                          algo,
                                          workspace,
                                          workspace_size,
                                          &beta,
                                          input_desc,
                                          input_contig.data_ptr<float>());  // dx is our input
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionBackwardData failed");

    // Add bias
    if (bias.numel() > 0) {
        status = cudnnAddTensor(cudnn_handle,
                                &alpha,
                                bias_desc,
                                bias_contig.data_ptr<float>(),
                                &alpha,  // beta = 1.0, accumulate
                                output_desc,
                                output.data_ptr<float>());
        TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnAddTensor failed");
    }

    // Cleanup
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    if (workspace) cudaFree(workspace);
    cudnnDestroy(cudnn_handle);

    return output;
}

// -----------------------------------------------------------------------------
// 2. Add bias forward
// -----------------------------------------------------------------------------
torch::Tensor add_bias_forward(torch::Tensor input, torch::Tensor bias) {
    check_cuda_float(input, "input");
    check_cuda_float(bias, "bias");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(bias.dim() == 2 && bias.size(1) == 1, "bias must be [channels, 1]");

    int channels = input.size(1);
    TORCH_CHECK(bias.size(0) == channels, "bias first dim must match channels");

    auto output = torch::empty_like(input);
    long long total_elements = input.numel();

    auto input_contig = input.contiguous();
    auto bias_contig = bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_bias_launcher(output.data_ptr<float>(),
                      input_contig.data_ptr<float>(),
                      bias_contig.data_ptr<float>(),
                      channels,
                      input.size(2),
                      total_elements,
                      stream);
    return output;
}

// -----------------------------------------------------------------------------
// 3. Roll forward
// -----------------------------------------------------------------------------
torch::Tensor roll_forward(torch::Tensor input, int64_t shifts, int64_t dims) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(dims == 2, "only dims=2 supported");
    TORCH_CHECK(shifts >= -input.size(2) && shifts <= input.size(2), "shifts out of range");

    auto output = torch::empty_like(input);
    long long total_elements = input.numel();

    auto input_contig = input.contiguous();
    int length = input.size(2);
    int shift_int = static_cast<int>(shifts);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    roll_launcher(output.data_ptr<float>(),
                  input_contig.data_ptr<float>(),
                  length,
                  shift_int,
                  total_elements,
                  stream);
    return output;
}

// -----------------------------------------------------------------------------
// 4. View forward (copy)
// -----------------------------------------------------------------------------
torch::Tensor view_forward(torch::Tensor input, int64_t out_features, int64_t num_features) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.size(0) * input.size(1) * input.size(2) == out_features * num_features,
                "view shape mismatch");

    auto output = torch::empty({input.size(0), num_features}, input.options());
    long long total_elements = input.numel();

    auto input_contig = input.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    view_launcher(output.data_ptr<float>(),
                  input_contig.data_ptr<float>(),
                  total_elements,
                  stream);
    return output;
}

// -----------------------------------------------------------------------------
// 5. Linear forward
// -----------------------------------------------------------------------------
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    check_cuda_float(input, "input");
    check_cuda_float(weight, "weight");
    check_cuda_float(bias, "bias");

    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int batch_size = input.size(0);
    int num_features = input.size(1);
    int out_features = weight.size(0);

    TORCH_CHECK(weight.size(1) == num_features, "weight width must match input features");
    TORCH_CHECK(bias.size(0) == out_features, "bias length must match output features");

    auto output = torch::empty({batch_size, out_features}, input.options());

    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    linear_launcher(output.data_ptr<float>(),
                    input_contig.data_ptr<float>(),
                    weight_contig.data_ptr<float>(),
                    bias_contig.data_ptr<float>(),
                    batch_size,
                    num_features,
                    out_features,
                    stream);
    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias_forward", &add_bias_forward,
          "Add bias forward", py::arg("input"), py::arg("bias"));
    m.def("roll_forward", &roll_forward,
          "Roll forward", py::arg("input"), py::arg("shifts"), py::arg("dims"));
    m.def("view_forward", &view_forward,
          "View forward", py::arg("input"), py::arg("out_features"), py::arg("num_features"));
    m.def("linear_forward", &linear_forward,
          "Linear forward", py::arg("input"), py::arg("weight"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
