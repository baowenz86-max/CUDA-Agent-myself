// binding.cpp for custom ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu (to be linked)
extern "C" void custom_adaptive_avg_pool1d_launcher(const float* input,
                                                    float* output,
                                                    int batch,
                                                    int channels,
                                                    int in_len,
                                                    int out_len,
                                                    cudaStream_t stream);

extern "C" void custom_slice_launcher(const float* input,
                                      float* output,
                                      int batch,
                                      int channels,
                                      int in_len,
                                      int dim,
                                      int start,
                                      int end,
                                      cudaStream_t stream);

extern "C" void custom_std_launcher(const float* input,
                                    float* output,
                                    int batch,
                                    int channels,
                                    int length,
                                    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentCudaStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// cuDNN convolution forward
torch::Tensor custom_conv1d(torch::Tensor input,
                            torch::Tensor weight,
                            c10::optional<torch::Tensor> bias_opt,
                            int64_t stride,
                            int64_t padding,
                            int64_t dilation,
                            int64_t groups) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (out_channels, in_channels/groups, kernel_size)");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->is_cuda(), "bias must be CUDA tensor");
        TORCH_CHECK(bias_opt->dtype() == torch::kFloat32, "bias must be float32");
        TORCH_CHECK(bias_opt->dim() == 1, "bias must be 1D");
    }

    cudnnHandle_t handle;
    cudnnCreate(&handle);
    cudaStream_t stream = getCurrentCudaStream();
    cudnnSetStream(handle, stream);

    // Input descriptor: [batch, in_channels, length]
    // cuDNN expects NCHW for 2D, but for 1D we use CUDNN_TENSOR_NCHW with only one spatial dimension (length=1 for height?)
    // Actually cuDNN supports 1D convolution via 2D with height=1.
    cudnnTensorDescriptor_t input_desc;
    cudnnCreateTensorDescriptor(&input_desc);
    int64_t in_dim[] = {input.size(0), input.size(1), input.size(2)};
    cudnnSetTensorNdDescriptor(input_desc, CUDNN_DATA_FLOAT, 3, in_dim, nullptr);

    // Weight descriptor: cuDNN filter descriptor for 2D convolution: [out_channels, in_channels/groups, kernel_size, 1]
    cudnnFilterDescriptor_t filter_desc;
    cudnnCreateFilterDescriptor(&filter_desc);
    int64_t filt_dim[] = {weight.size(0), weight.size(1), weight.size(2), 1};
    cudnnSetFilterNdDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, filt_dim);

    // Convolution descriptor
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnCreateConvolutionDescriptor(&conv_desc);
    int pad[] = {static_cast<int>(padding), 0};
    int stride_arr[] = {static_cast<int>(stride), 1};
    int dilation_arr[] = {static_cast<int>(dilation), 1};
    cudnnSetConvolutionNdDescriptor(conv_desc, 2, pad, stride_arr, dilation_arr, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetConvolutionGroupCount(conv_desc, static_cast<int>(groups));

    // Output dimensions
    int64_t out_len = (input.size(2) + 2 * padding - dilation * (weight.size(2) - 1) - 1) / stride + 1;
    auto output = torch::empty({input.size(0), weight.size(0), out_len}, input.options());

    cudnnTensorDescriptor_t output_desc;
    cudnnCreateTensorDescriptor(&output_desc);
    int64_t out_dim[] = {output.size(0), output.size(1), output.size(2)};
    cudnnSetTensorNdDescriptor(output_desc, CUDNN_DATA_FLOAT, 3, out_dim, nullptr);

    // Find algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm(handle, input_desc, filter_desc, conv_desc, output_desc,
                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo);

    size_t workspace_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(handle, input_desc, filter_desc, conv_desc, output_desc, algo, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    if (bias_opt.has_value()) {
        // Use bias as beta, but cuDNN adds bias after convolution.
        // For bias, we can use cudnnConvolutionBiasActivationForward or simply set beta=1 and add bias pre-filled.
        // Simpler: first do convolution without bias, then add bias manually in a separate kernel, but we avoid kernels.
        // Instead, use cudnnConvolutionForward with pre-computed bias as beta? Not directly supported.
        // Common approach: allocate output, fill with bias, then call with beta=1, alpha=1.
        // We'll use cudaMemcpyAsync to initialize output with bias.
        auto bias_ptr = bias_opt->data_ptr<float>();
        for (int i = 0; i < output.size(0); ++i) {
            cudaMemcpyAsync(output.data_ptr<float>() + i * output.size(1) * out_len,
                            bias_ptr, output.size(1) * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
        beta = 1.0f;
    }
    cudnnConvolutionForward(handle, &alpha, input_desc, input.data_ptr<float>(),
                            filter_desc, weight.data_ptr<float>(), conv_desc, algo,
                            workspace, workspace_size, &beta, output_desc, output.data_ptr<float>());

    if (workspace) cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(handle);

    return output;
}

// Adaptive average pooling wrapper
torch::Tensor custom_adaptive_avg_pool1d(torch::Tensor input, int64_t output_size) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");

    auto output = torch::empty({input.size(0), input.size(1), output_size}, input.options());

    cudaStream_t stream = getCurrentCudaStream();
    custom_adaptive_avg_pool1d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        input.size(2),
        static_cast<int>(output_size),
        stream
    );
    return output;
}

// Slice wrapper (only dim=2 supported)
torch::Tensor custom_slice(torch::Tensor input, int64_t dim, int64_t start, int64_t end) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dim == 2, "only dim=2 slice is supported");
    TORCH_CHECK(start >= 0 && end <= input.size(2) && start < end, "invalid slice range");

    auto output = torch::empty({input.size(0), input.size(1), end - start}, input.options());

    cudaStream_t stream = getCurrentCudaStream();
    custom_slice_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        input.size(2),
        static_cast<int>(dim),
        static_cast<int>(start),
        static_cast<int>(end),
        stream
    );
    return output;
}

// Std reduction wrapper
torch::Tensor custom_std(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(dim == 2, "only dim=2 reduction is supported");

    auto output = torch::empty({input.size(0), input.size(1)}, input.options());

    cudaStream_t stream = getCurrentCudaStream();
    custom_std_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        input.size(0),
        input.size(1),
        input.size(2),
        stream
    );
    return output;
}

// Register all functions
void register_custom_ops(pybind11::module& m) {
    m.def("custom_conv1d", &custom_conv1d,
          "Custom 1D convolution using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("dilation"), py::arg("groups"));
    m.def("custom_adaptive_avg_pool1d", &custom_adaptive_avg_pool1d,
          "Custom adaptive average pooling 1D",
          py::arg("input"), py::arg("output_size"));
    m.def("custom_slice", &custom_slice,
          "Custom slice along dimension",
          py::arg("input"), py::arg("dim"), py::arg("start"), py::arg("end"));
    m.def("custom_std", &custom_std,
          "Custom standard deviation reduction",
          py::arg("input"), py::arg("dim"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);