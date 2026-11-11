// custom_ops_binding.cpp
// PyTorch bindings for custom_stft, custom_abs, custom_remainder, custom_conv1d.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <ATen/cuda/CUDAContext.h>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void custom_stft_launcher(
    const float* input,
    long long batch,
    long long length,
    long long n_fft,
    long long hop_length,
    float* output_real,
    float* output_imag,
    cudaStream_t stream);

extern "C" void custom_abs_launcher(
    const float* input_real,
    const float* input_imag,
    float* output,
    long long numel,
    long long stride_real_elem,
    long long stride_imag_elem,
    long long output_stride_elem,
    cudaStream_t stream);

extern "C" void custom_remainder_launcher(
    const float* input,
    float* output,
    float divisor,
    long long numel,
    long long input_stride_elem,
    long long output_stride_elem,
    cudaStream_t stream);

// Helper: check CUDA error
static inline void check_cuda_error(cudaError_t err, const char* msg) {
    TORCH_CHECK(err == cudaSuccess, msg, ": ", cudaGetErrorString(err));
}

// Helper: check cuDNN error
static inline void check_cudnn_error(cudnnStatus_t status, const char* msg) {
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, msg, ": ", cudnnGetErrorString(status));
}

// custom_stft wrapper
// Input x: float32 CUDA tensor [batch, length]
// Output: complex64 CUDA tensor [batch, n_freq, frames] with non-contiguous strides
// matching reference: [n_freq*frames, 1, n_freq]
torch::Tensor custom_stft(torch::Tensor x, int64_t n_fft, int64_t hop_length) {
    TORCH_CHECK(x.is_cuda(), "custom_stft: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_stft: input must be float32");
    TORCH_CHECK(x.dim() == 2, "custom_stft: input must be 2D");

    // Ensure input is contiguous for launcher
    auto x_contig = x.contiguous();
    long long batch = x_contig.size(0);
    long long length = x_contig.size(1);
    long long n_freq = n_fft / 2 + 1;
    long long frames = 1 + length / hop_length;

    // Allocate output with the exact strides from reference: [n_freq*frames, 1, n_freq]
    auto output = torch::empty_strided(
        {batch, n_freq, frames},
        {n_freq * frames, 1, n_freq},
        torch::dtype(torch::kComplexFloat).device(x.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get pointers to real and imag parts of output complex tensor
    float* output_real = reinterpret_cast<float*>(output.data_ptr<c10::complex<float>>());
    float* output_imag = output_real + 1; // interleaved layout; second float is imaginary

    // call launcher
    custom_stft_launcher(
        x_contig.data_ptr<float>(),
        batch,
        length,
        n_fft,
        hop_length,
        output_real,
        output_imag,
        stream);

    // check for kernel errors
    cudaError_t err = cudaGetLastError();
    check_cuda_error(err, "custom_stft kernel launch failed");

    return output;
}

// custom_abs wrapper
// Input x: complex64 CUDA tensor [batch, freq, frames] (may be non-contiguous)
// Output: float32 CUDA tensor with same shape (contiguous)
torch::Tensor custom_abs(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "custom_abs: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kComplexFloat, "custom_abs: input must be complex64");
    TORCH_CHECK(x.dim() == 3, "custom_abs: input must be 3D");

    // Get shape
    long long batch = x.size(0);
    long long freq = x.size(1);
    long long frames = x.size(2);
    long long numel = x.numel();

    // Compute strides in elements for real and imaginary pointers.
    // For complex64 tensor, data_ptr<c10::complex<float>> points to interleaved real/imag pairs.
    // The real part pointer is cast from complex data pointer, and the imag pointer is one float offset.
    // We need element strides for real and imag assuming they are consecutive floats.
    // Let's derive: If tensor stride for complex dimension d is s_d (in elements of complex), then
    // real part offset increment for that dimension is 2*s_d (floats), imag similarly.
    auto strides = x.strides(); // in complex elements
    long long stride0 = strides[0];
    long long stride1 = strides[1];
    long long stride2 = strides[2];

    // For real and imag, we treat them as separate float arrays with strides doubled.
    long long stride_real_elem_0 = stride0 * 2;
    long long stride_real_elem_1 = stride1 * 2;
    long long stride_real_elem_2 = stride2 * 2;
    // For imag, offset by one float
    long long stride_imag_elem_0 = stride0 * 2;
    long long stride_imag_elem_1 = stride1 * 2;
    long long stride_imag_elem_2 = stride2 * 2;

    // Since our launcher only supports a single stride for real and imag (elem_stride_real, elem_stride_imag),
    // we need to adapt. The current launcher assumes that the linear index maps to offset = idx * stride_real_elem.
    // That only works if the tensor is contiguous with stride_real=1. For non-contiguous we need a different approach.
    // To simplify, we can make the input contiguous, then real/imag are interleaved with stride 1 for both.
    auto x_contig = x.contiguous();
    float* input_real = reinterpret_cast<float*>(x_contig.data_ptr<c10::complex<float>>());
    float* input_imag = input_real + 1;

    // Output is contiguous float32 with same shape
    auto output = torch::empty({batch, freq, frames}, torch::dtype(torch::kFloat32).device(x.device()));
    float* out_ptr = output.data_ptr<float>();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // For contiguous complex, real stride = 2 (since each complex element is 2 floats), imag stride = 2 as well, starting at offset 1.
    // But our launcher expects elem_stride_real and elem_stride_imag as the distance in elements between consecutive real parts.
    // For contiguous complex with interleaved layout, real parts are 2 floats apart, so stride_real_elem = 2.
    long long stride_real_elem = 2;
    long long stride_imag_elem = 2;
    long long output_stride_elem = 1; // output contiguous

    custom_abs_launcher(
        input_real,
        input_imag,
        out_ptr,
        numel,
        stride_real_elem,
        stride_imag_elem,
        output_stride_elem,
        stream);

    cudaError_t err = cudaGetLastError();
    check_cuda_error(err, "custom_abs kernel launch failed");

    return output;
}

// custom_remainder wrapper
// Input x: float32 CUDA tensor [batch, freq, frames] (may be non-contiguous)
// Output: float32 CUDA tensor with same shape (contiguous)
torch::Tensor custom_remainder(torch::Tensor x, double divisor) {
    TORCH_CHECK(x.is_cuda(), "custom_remainder: input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_remainder: input must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_remainder: input must be 3D");

    auto x_contig = x.contiguous();
    long long numel = x_contig.numel();

    auto output = torch::empty_like(x_contig, x_contig.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_remainder_launcher(
        x_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<float>(divisor),
        numel,
        1, // input contiguous stride
        1, // output contiguous stride
        stream);

    cudaError_t err = cudaGetLastError();
    check_cuda_error(err, "custom_remainder kernel launch failed");

    return output;
}

// custom_conv1d wrapper using cuDNN
// Input x: float32 CUDA tensor [batch, in_channels, length_in] (may be non-contiguous)
// weight: float32 CUDA tensor [out_channels, in_channels, kernel_size]
// bias: float32 CUDA tensor [out_channels] (may be None)
// stride, padding: ints
// Output: float32 CUDA tensor [batch, out_channels, length_out] contiguous
torch::Tensor custom_conv1d(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int64_t stride, int64_t padding) {
    TORCH_CHECK(x.is_cuda(), "custom_conv1d: input must be CUDA");
    TORCH_CHECK(weight.is_cuda(), "custom_conv1d: weight must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "custom_conv1d: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "custom_conv1d: weight must be float32");
    TORCH_CHECK(x.dim() == 3, "custom_conv1d: input must be 3D");
    TORCH_CHECK(weight.dim() == 3, "custom_conv1d: weight must be 3D");

    long long batch = x.size(0);
    long long in_channels = x.size(1);
    long long length_in = x.size(2);
    long long out_channels = weight.size(0);
    long long kernel_size = weight.size(2);
    long long length_out = (length_in + 2 * padding - kernel_size) / stride + 1;

    // Ensure contiguous
    auto x_contig = x.contiguous();
    auto w_contig = weight.contiguous();
    // Bias may be None; handle later

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    cudnnStatus_t status = cudnnCreate(&cudnn);
    check_cudnn_error(status, "cudnnCreate failed");

    // Set stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    status = cudnnSetStream(cudnn, stream);
    check_cudnn_error(status, "cudnnSetStream failed");

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc;
    cudnnTensorDescriptor_t y_desc;
    cudnnTensorDescriptor_t b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // For cuDNN, treat 1D conv as 2D conv with height=1.
    int dimA[4] = {static_cast<int>(batch), static_cast<int>(in_channels), 1, static_cast<int>(length_in)};
    int strideA[4] = {static_cast<int>(in_channels * length_in), static_cast<int>(length_in), static_cast<int>(length_in), 1};
    cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 4, dimA, strideA);

    int dimY[4] = {static_cast<int>(batch), static_cast<int>(out_channels), 1, static_cast<int>(length_out)};
    int strideY[4] = {static_cast<int>(out_channels * length_out), static_cast<int>(length_out), static_cast<int>(length_out), 1};
    cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 4, dimY, strideY);

    // Filter descriptor: filter layout [out_channels, in_channels, 1, kernel_size]
    int dimW[4] = {static_cast<int>(out_channels), static_cast<int>(in_channels), 1, static_cast<int>(kernel_size)};
    cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, dimW);

    // Convolution descriptor
    int padA[2] = {0, static_cast<int>(padding)};
    int strideAconv[2] = {1, static_cast<int>(stride)};
    int dilA[2] = {1, 1};
    cudnnSetConvolutionNdDescriptor(conv_desc, 2, padA, strideAconv, dilA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    // Bias descriptor
    int dimB[4] = {1, static_cast<int>(out_channels), 1, 1};
    int strideB[4] = {static_cast<int>(out_channels), 1, 1, 1};
    cudnnSetTensorNdDescriptor(b_desc, CUDNN_DATA_FLOAT, 4, dimB, strideB);

    // Allocate output
    auto output = torch::empty({batch, out_channels, length_out}, torch::dtype(torch::kFloat32).device(x.device()));
    float* y_ptr = output.data_ptr<float>();

    // Convolution forward algorithm
    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, x_desc, w_desc, conv_desc, y_desc, 1, 0, &algo);

    // Workspace size
    size_t ws_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size);
    void* ws_ptr = nullptr;
    if (ws_size > 0) {
        cudaMalloc(&ws_ptr, ws_size);
        if (cudaGetLastError() != cudaSuccess) {
            // handle error
        }
    }

    float alpha = 1.0f, beta = 0.0f;
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32, "bias must be CUDA float32");
        auto b_contig = bias.contiguous();
        beta = 1.0f; // add bias after convolution
        status = cudnnConvolutionForward(
            cudnn,
            &alpha,
            x_desc, x_contig.data_ptr<float>(),
            w_desc, w_contig.data_ptr<float>(),
            conv_desc, algo, ws_ptr, ws_size,
            &beta,
            y_desc, y_ptr);
        check_cudnn_error(status, "cudnnConvolutionForward failed with bias");
        // Add bias: cuDNN adds bias if beta=1 and b_desc set, but we need to call cudnnAddTensor or set bias mode.
        // Simpler: call cudnnAddTensor after convolution.
        // Actually above with beta=1 and no bias descriptor set, it would add 1*y, not bias. Need to use bias descriptor.
        // Let's use cudnnConvolutionBiasActivationForward or separate cudnnAddTensor.
        // We'll do separate: first conv with beta=0, then add bias.
        // So reset beta=0, then add bias.
        // Redo:
        status = cudnnConvolutionForward(
            cudnn,
            &alpha,
            x_desc, x_contig.data_ptr<float>(),
            w_desc, w_contig.data_ptr<float>(),
            conv_desc, algo, ws_ptr, ws_size,
            &beta0 /* 0 */,
            y_desc, y_ptr);
        check_cudnn_error(status, "cudnnConvolutionForward failed");
        // Add bias
        float alpha_bias = 1.0f, beta_bias = 1.0f;
        status = cudnnAddTensor(cudnn, &alpha_bias, b_desc, b_contig.data_ptr<float>(), &beta_bias, y_desc, y_ptr);
        check_cudnn_error(status, "cudnnAddTensor failed");
    } else {
        beta = 0.0f;
        status = cudnnConvolutionForward(
            cudnn,
            &alpha,
            x_desc, x_contig.data_ptr<float>(),
            w_desc, w_contig.data_ptr<float>(),
            conv_desc, algo, ws_ptr, ws_size,
            &beta,
            y_desc, y_ptr);
        check_cudnn_error(status, "cudnnConvolutionForward failed");
    }

    // Free workspace
    if (ws_size > 0) cudaFree(ws_ptr);

    // Destroy descriptors and handle
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("custom_stft", &custom_stft, "Custom STFT",
          py::arg("x"), py::arg("n_fft"), py::arg("hop_length"));
    m.def("custom_abs", &custom_abs, "Custom abs");
    m.def("custom_remainder", &custom_remainder, "Custom remainder",
          py::arg("x"), py::arg("divisor"));
    m.def("custom_conv1d", &custom_conv1d, "Custom conv1d using cuDNN",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("stride"), py::arg("padding"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
