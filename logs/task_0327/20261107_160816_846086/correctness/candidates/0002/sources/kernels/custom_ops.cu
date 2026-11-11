// custom_ops.cu
// CUDA kernels and launchers for custom_stft, custom_abs, custom_remainder, custom_conv1d.
// custom_conv1d uses cuDNN, included here.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cudnn.h>
#include <complex>

// Forward declarations for helper kernels
__global__ void pad_reflect_kernel(const float* __restrict__ input,
                                   float* __restrict__ padded,
                                   long long batch,
                                   long long length,
                                   int pad_left,
                                   int pad_right,
                                   long long pad_length);

__global__ void extract_frames_kernel(const float* __restrict__ padded,
                                      float* __restrict__ frames,
                                      long long batch,
                                      long long num_frames,
                                      long long n_fft,
                                      long long hop_length,
                                      long long pad_length);

__global__ void scatter_fft_kernel(const float2* __restrict__ fft_out,
                                   float* __restrict__ output_real,
                                   float* __restrict__ output_imag,
                                   long long batch,
                                   long long n_freq,
                                   long long frames);


// Helper: check CUDA errors
static void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        // No-op; binding will check after launch.
    }
}

// -----------------------------------------------------------------------------
// Elementwise kernels for custom_abs and custom_remainder.
// -----------------------------------------------------------------------------
__global__ void abs_kernel(const float* __restrict__ input_real,
                           const float* __restrict__ input_imag,
                           float* __restrict__ output,
                           long long numel,
                           long long elem_stride_real,
                           long long elem_stride_imag) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    long long real_offset = idx * elem_stride_real;
    long long imag_offset = idx * elem_stride_imag;
    float re = input_real[real_offset];
    float im = input_imag[imag_offset];
    output[idx] = sqrtf(re * re + im * im);
}

__global__ void remainder_kernel(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  float divisor,
                                  long long numel,
                                  long long stride) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (idx >= numel) return;
    long long offset = idx * stride;
    float val = input[offset];
    float r = fmodf(val, divisor);
    if (r != 0 && ((r < 0 && divisor > 0) || (r > 0 && divisor < 0))) {
        r += divisor;
    }
    output[idx] = r;
}

// -----------------------------------------------------------------------------
// Helper kernels used by STFT
// -----------------------------------------------------------------------------

// Pad input with reflect padding
__global__ void pad_reflect_kernel(const float* __restrict__ input,
                                   float* __restrict__ padded,
                                   long long batch,
                                   long long length,
                                   int pad_left,
                                   int pad_right,
                                   long long pad_length)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = batch * pad_length;
    if (idx >= total) return;
    long long b = idx / pad_length;
    long long pos = idx % pad_length;
    long long src_idx;
    if (pos < pad_left) {
        src_idx = pad_left - pos;
    } else if (pos >= pad_left + length) {
        long long k = pos - (pad_left + length);
        src_idx = length - 2 - k;
    } else {
        src_idx = pos - pad_left;
    }
    padded[idx] = input[b * length + src_idx];
}

// Extract frames from padded input
__global__ void extract_frames_kernel(const float* __restrict__ padded,
                                      float* __restrict__ frames,
                                      long long batch,
                                      long long num_frames,
                                      long long n_fft,
                                      long long hop_length,
                                      long long pad_length)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = batch * num_frames * n_fft;
    if (idx >= total) return;
    long long t = idx % n_fft;
    long long tmp = idx / n_fft;
    long long f = tmp % num_frames;
    long long b = tmp / num_frames;
    long long src_pos = f * hop_length + t;
    frames[idx] = padded[b * pad_length + src_pos];
}

// Scatter FFT output into output arrays with strides
__global__ void scatter_fft_kernel(const float2* __restrict__ fft_out,
                                   float* __restrict__ output_real,
                                   float* __restrict__ output_imag,
                                   long long batch,
                                   long long n_freq,
                                   long long frames)
{
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = batch * n_freq * frames;
    if (idx >= total) return;
    long long frame = idx % frames;
    long long tmp = idx / frames;
    long long freq = tmp % n_freq;
    long long b = tmp / n_freq;
    long long fft_idx = (b * frames + frame) * n_freq + freq;
    float2 val = fft_out[fft_idx];
    long long out_idx = b * n_freq * frames + freq + frame * n_freq;
    output_real[out_idx] = val.x;
    output_imag[out_idx] = val.y;
}

// -----------------------------------------------------------------------------
// STFT launcher using cuFFT.
// Input: x [batch, length] float32, assumed contiguous (binding ensures contiguous).
// Output: complex64 [batch, n_freq, frames] with strides [n_freq*frames, 1, n_freq].
// -----------------------------------------------------------------------------
extern "C" void custom_stft_launcher(
    const float* input,
    long long batch,
    long long length,
    long long n_fft,
    long long hop_length,
    float* output_real,
    float* output_imag,
    cudaStream_t stream)
{
    long long n_freq = n_fft / 2 + 1;
    long long frames = 1 + length / hop_length;

    // cuFFT plan for many frames
    int rank = 1;
    int n[1] = {static_cast<int>(n_fft)};
    int inembed[1] = {1};
    int onembed[1] = {1};
    int istride = 1;
    int ostride = 1;
    int dist_in = static_cast<int>(n_fft);  // distance between frames after extraction
    int dist_out = static_cast<int>(n_freq); // distance between output frames

    cufftHandle plan_frames;
    cufftResult res;
    res = cufftPlanMany(&plan_frames, rank, n,
                        inembed, istride, dist_in,
                        onembed, ostride, dist_out,
                        CUFFT_R2C, static_cast<int>(batch * frames));
    if (res != CUFFT_SUCCESS) return;
    cufftSetStream(plan_frames, stream);

    // Allocate padded input buffer
    long long pad_left = n_fft / 2;
    long long pad_right = n_fft / 2;
    long long pad_length = length + pad_left + pad_right;
    float* padded_input;
    cudaMalloc(&padded_input, pad_length * batch * sizeof(float));
    cudaMemsetAsync(padded_input, 0, pad_length * batch * sizeof(float), stream);

    int threads = 256;
    int blocks_pad = (pad_length * batch + threads - 1) / threads;
    pad_reflect_kernel<<<blocks_pad, threads, 0, stream>>>(
        input, padded_input, batch, length, static_cast<int>(pad_left), static_cast<int>(pad_right), pad_length);

    // Allocate frame buffer: [batch * frames, n_fft] contiguous
    long long total_frames = batch * frames;
    long long frame_input_size = total_frames * n_fft;
    float* frame_buffer;
    cudaMalloc(&frame_buffer, frame_input_size * sizeof(float));

    int blocks_extract = (frame_input_size + threads - 1) / threads;
    extract_frames_kernel<<<blocks_extract, threads, 0, stream>>>(
        padded_input, frame_buffer, batch, frames, n_fft, hop_length, pad_length);

    // Allocate FFT output complex buffer
    cufftComplex* fft_out;
    cudaMalloc(&fft_out, total_frames * n_freq * sizeof(cufftComplex));

    res = cufftExecR2C(plan_frames, frame_buffer, fft_out);
    if (res != CUFFT_SUCCESS) {
        cudaFree(padded_input);
        cudaFree(frame_buffer);
        cudaFree(fft_out);
        cufftDestroy(plan_frames);
        return;
    }

    // Scatter to output with required strides
    long long total_out_elements = batch * n_freq * frames;
    int blocks_scatter = (total_out_elements + threads - 1) / threads;
    scatter_fft_kernel<<<blocks_scatter, threads, 0, stream>>>(
        (const float2*)fft_out, output_real, output_imag, batch, n_freq, frames);

    cudaFree(padded_input);
    cudaFree(frame_buffer);
    cudaFree(fft_out);
    cufftDestroy(plan_frames);
}

// -----------------------------------------------------------------------------
// Custom abs launcher: input complex64 tensor (contiguous after binding)
// -----------------------------------------------------------------------------
extern "C" void custom_abs_launcher(
    const float* input_real,
    const float* input_imag,
    float* output,
    long long numel,
    long long stride_real_elem,
    long long stride_imag_elem,
    long long output_stride_elem,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    abs_kernel<<<blocks, threads, 0, stream>>>(
        input_real, input_imag, output, numel,
        stride_real_elem, stride_imag_elem);
}

// -----------------------------------------------------------------------------
// Custom remainder launcher: input float32 (contiguous after binding)
// -----------------------------------------------------------------------------
extern "C" void custom_remainder_launcher(
    const float* input,
    float* output,
    float divisor,
    long long numel,
    long long input_stride_elem,
    long long output_stride_elem,
    cudaStream_t stream)
{
    int threads = 256;
    long long blocks = (numel + threads - 1) / threads;
    remainder_kernel<<<blocks, threads, 0, stream>>>(
        input, output, divisor, numel, input_stride_elem);
}

// -----------------------------------------------------------------------------
// Custom conv1d launcher using cuDNN
// -----------------------------------------------------------------------------
extern "C" void custom_conv1d_launcher(
    const float* input,           // [batch, in_channels, length_in]
    const float* weight,          // [out_channels, in_channels, kernel_size]
    const float* bias,            // [out_channels] may be nullptr
    float* output,                // [batch, out_channels, length_out]
    long long batch,
    long long in_channels,
    long long length_in,
    long long out_channels,
    long long kernel_size,
    long long stride,
    long long padding,
    cudaStream_t stream)
{
    long long length_out = (length_in + 2 * padding - kernel_size) / stride + 1;

    cudnnHandle_t cudnn;
    cudnnStatus_t status = cudnnCreate(&cudnn);
    if (status != CUDNN_STATUS_SUCCESS) return;
    cudnnSetStream(cudnn, stream);

    cudnnTensorDescriptor_t x_desc, y_desc, b_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    int dimA[4] = {static_cast<int>(batch), static_cast<int>(in_channels), 1, static_cast<int>(length_in)};
    int strideA[4] = {static_cast<int>(in_channels * length_in), static_cast<int>(length_in), static_cast<int>(length_in), 1};
    cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 4, dimA, strideA);

    int dimY[4] = {static_cast<int>(batch), static_cast<int>(out_channels), 1, static_cast<int>(length_out)};
    int strideY[4] = {static_cast<int>(out_channels * length_out), static_cast<int>(length_out), static_cast<int>(length_out), 1};
    cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 4, dimY, strideY);

    int dimW[4] = {static_cast<int>(out_channels), static_cast<int>(in_channels), 1, static_cast<int>(kernel_size)};
    cudnnSetFilterNdDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, dimW);

    int padA[2] = {0, static_cast<int>(padding)};
    int strideAconv[2] = {1, static_cast<int>(stride)};
    int dilA[2] = {1, 1};
    cudnnSetConvolutionNdDescriptor(conv_desc, 2, padA, strideAconv, dilA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int dimB[4] = {1, static_cast<int>(out_channels), 1, 1};
    int strideB[4] = {static_cast<int>(out_channels), 1, 1, 1};
    cudnnSetTensorNdDescriptor(b_desc, CUDNN_DATA_FLOAT, 4, dimB, strideB);

    cudnnConvolutionFwdAlgo_t algo;
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn, x_desc, w_desc, conv_desc, y_desc, 1, 0, &algo);

    size_t ws_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, x_desc, w_desc, conv_desc, y_desc, algo, &ws_size);
    void* ws_ptr = nullptr;
    if (ws_size > 0) {
        cudaMalloc(&ws_ptr, ws_size);
    }

    float alpha = 1.0f, beta = 0.0f;
    status = cudnnConvolutionForward(cudnn, &alpha, x_desc, input, w_desc, weight,
                                     conv_desc, algo, ws_ptr, ws_size, &beta, y_desc, output);
    if (status != CUDNN_STATUS_SUCCESS) {
        cudaFree(ws_ptr);
        cudnnDestroyTensorDescriptor(x_desc);
        cudnnDestroyTensorDescriptor(y_desc);
        cudnnDestroyTensorDescriptor(b_desc);
        cudnnDestroyFilterDescriptor(w_desc);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroy(cudnn);
        return;
    }

    if (bias != nullptr) {
        float alpha_bias = 1.0f, beta_bias = 1.0f;
        status = cudnnAddTensor(cudnn, &alpha_bias, b_desc, bias, &beta_bias, y_desc, output);
    }

    if (ws_size > 0) cudaFree(ws_ptr);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
}
