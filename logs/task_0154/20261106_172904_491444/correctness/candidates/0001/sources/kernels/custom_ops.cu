// kernels/custom_ops.cu
// CUDA kernels and launchers for custom ops, no torch dependencies.
// Conv2d uses cuDNN; other ops use custom kernels or cuFFT.

#include <cuda_runtime.h>
#include <cufft.h>
#include <cudnn.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define CUDNN_CHECK(ans) { cudnnAssert((ans), __FILE__, __LINE__); }
inline void cudnnAssert(cudnnStatus_t status, const char *file, int line, bool abort=true) {
    if (status != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "cuDNN error: %s %s %d\n", cudnnGetErrorString(status), file, line);
        if (abort) exit(status);
    }
}

#define CUFFT_CHECK(ans) { cufftAssert((ans), __FILE__, __LINE__); }
inline void cufftAssert(cufftResult_t result, const char *file, int line, bool abort=true) {
    if (result != CUFFT_SUCCESS) {
        fprintf(stderr, "cuFFT error: %d %s %d\n", result, file, line);
        if (abort) exit(result);
    }
}

// -----------------------------------------------------------------------------
// STFT kernel: computes short-time Fourier transform.
// Input: float* input: [batch, sequence_length]
// Output: cuComplex* output: [batch, freq_bins, frames] (complex64)
// The launcher uses cuFFT for each frame.
// -----------------------------------------------------------------------------
extern "C" void stft_forward_launcher(
    const float* input,
    cuComplex* output,
    int batch,
    int sequence_length,
    int n_fft,
    int win_length,
    int hop_length,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// view_as_real_permute kernel: converts complex input to two-channel float output
// with permute(0,3,1,2).
// Input: const cuComplex* input: [batch, freq_bins, frames]
// Output: float* output: [batch, 2, freq_bins, frames]
// -----------------------------------------------------------------------------
__global__ void view_as_real_permute_kernel(
    const cuComplex* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int freq_bins,
    int frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * freq_bins * frames;
    if (idx < total) {
        int f = idx % frames;
        int fb = (idx / frames) % freq_bins;
        int b = idx / (frames * freq_bins);

        cuComplex val = input[b * freq_bins * frames + fb * frames + f];
        // output layout: [batch, 2, freq_bins, frames]
        // real part at [b,0,fb,f], imag at [b,1,fb,f]
        output[b * 2 * freq_bins * frames + 0 * freq_bins * frames + fb * frames + f] = val.x;
        output[b * 2 * freq_bins * frames + 1 * freq_bins * frames + fb * frames + f] = val.y;
    }
}

extern "C" void view_as_real_permute_launcher(
    const cuComplex* input,
    float* output,
    int batch,
    int freq_bins,
    int frames,
    cudaStream_t stream
) {
    int total = batch * freq_bins * frames;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    view_as_real_permute_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, freq_bins, frames
    );
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// index_select kernel: selects slices along given dim (supports dim=2 only for now).
// Input: const float* input: [batch, channels, in_dim, frames]
//        const int64_t* indices: [num_indices]
// Output: float* output: [batch, channels, num_indices, frames]
// -----------------------------------------------------------------------------
__global__ void index_select_dim2_kernel(
    const float* __restrict__ input,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_dim,
    int frames,
    int num_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * num_indices * frames;
    if (idx < total) {
        int f = idx % frames;
        int ni = (idx / frames) % num_indices;
        int c = (idx / (frames * num_indices)) % channels;
        int b = idx / (frames * num_indices * channels);

        int64_t src_index = indices[ni];
        // assume valid index
        float val = input[(b * channels + c) * in_dim * frames + src_index * frames + f];
        output[((b * channels + c) * num_indices + ni) * frames + f] = val;
    }
}

extern "C" void index_select_forward_launcher(
    const float* input,
    const int64_t* indices,
    float* output,
    int batch,
    int channels,
    int in_dim,
    int frames,
    int num_indices,
    int dim,
    cudaStream_t stream
) {
    if (dim != 2) {
        fprintf(stderr, "index_select kernel currently supports dim=2 only, got dim=%d\n", dim);
        exit(1);
    }
    int total = batch * channels * num_indices * frames;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    index_select_dim2_kernel<<<blocks, threads, 0, stream>>>(
        input, indices, output, batch, channels, in_dim, frames, num_indices
    );
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------------------------------------------------------
// STFT implementation using cuFFT
// -----------------------------------------------------------------------------
// Helper kernel to apply window and center padding/reflection.
// We'll do the padding in a kernel that copies input to a padded buffer.
__global__ void stft_pad_kernel(
    const float* __restrict__ input,
    float* __restrict__ padded,
    int sequence_length,
    int pad_left,
    int pad_right,
    int n_fft
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int padded_len = sequence_length + pad_left + pad_right;
    int total = padded_len;
    if (idx < total) {
        int src_idx = idx - pad_left;
        float val;
        if (src_idx >= 0 && src_idx < sequence_length) {
            val = input[src_idx];
        } else if (src_idx < 0) {
            // reflect left
            val = input[-src_idx]; // since reflection at boundary, index -src_idx
        } else {
            // reflect right
            int reflect_idx = sequence_length - 1 - (src_idx - sequence_length);
            if (reflect_idx < 0) reflect_idx = 0;
            val = input[reflect_idx];
        }
        padded[idx] = val;
    }
}

// STFT launcher
void stft_forward_launcher(
    const float* input,
    cuComplex* output,
    int batch,
    int sequence_length,
    int n_fft,
    int win_length,
    int hop_length,
    cudaStream_t stream
) {
    int freq_bins = n_fft / 2 + 1;
    int pad_left = n_fft / 2;
    int pad_right = n_fft - pad_left;
    int padded_len = sequence_length + pad_left + pad_right;
    int frames = 1 + (padded_len - n_fft) / hop_length; // center=True

    // Each frame is n_fft points; we need a temporary workspace for each batch frame
    // We'll allocate a per-batch workspace of size n_fft * frames (float) for real input after windowing
    // Also need complex spectrum for n_fft (or freq_bins)

    // For simplicity, we call cuFFT for each frame per batch, using a separate stream? Use current stream.
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftSetStream(plan, stream));
    CUFFT_CHECK(cufftPlan1d(&plan, n_fft, CUFFT_R2C, batch * frames)); // batch dimension separate? Actually plan for many
    // But we need to handle window and center padding.

    // Approach: allocate device memory for padded input per batch, then extract frames into a contiguous buffer of length n_fft * frames, apply window, then batched FFT.
    // To save memory, we can process one batch at a time.

    float *d_padded, *d_windowed_frames;
    cuComplex *d_spectrum;

    CUDA_CHECK(cudaMalloc(&d_padded, padded_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_windowed_frames, frames * n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, frames * freq_bins * sizeof(cuComplex)));

    // Create Hann window on device? For simplicity, generate on host and copy, or compute in kernel.
    // We'll create window on device via kernel that computes Hann each time.
    // But window size = win_length (<= n_fft). If win_length < n_fft, pad zeros in middle.
    // We'll compute window in kernel when applying.

    for (int b = 0; b < batch; ++b) {
        const float* input_b = input + b * sequence_length;

        // 1. Pad with reflection
        int total_pad = padded_len;
        int threads = 256;
        int blocks = (total_pad + threads - 1) / threads;
        stft_pad_kernel<<<blocks, threads, 0, stream>>>(
            input_b, d_padded, sequence_length, pad_left, pad_right, n_fft
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. Extract frames and apply window
        // We'll use a kernel to fill d_windowed_frames (frames * n_fft)
        // Kernel: for each frame f and sample i within n_fft, compute window value and multiply with padded value.
        int total_frames_samples = frames * n_fft;
        blocks = (total_frames_samples + threads - 1) / threads;
        // Use a kernel to perform this.
        // We'll define a lambda-free kernel below.
        // But to avoid code duplication, define another global kernel outside.
        // For now, we'll inline via a separate kernel defined earlier or later.
        // We'll call stft_frames_kernel defined below.
        // We'll define it after this function.
        stft_frames_kernel<<<blocks, threads, 0, stream>>>(
            d_padded, d_windowed_frames, frames, n_fft, hop_length, win_length
        );
        CUDA_CHECK(cudaGetLastError());

        // 3. Batched cuFFT R2C on frames
        CUFFT_CHECK(cufftExecR2C(plan, d_windowed_frames, d_spectrum));

        // 4. Copy spectrum (freq_bins per frame) to output slice [b, freq_bins, frames]
        // Output layout: [batch, freq_bins, frames] contiguous.
        // We need to transpose from [frames, freq_bins] to [freq_bins, frames].
        // We can use a kernel to copy with transpose.
        int total_out = freq_bins * frames;
        blocks = (total_out + threads - 1) / threads;
        stft_copy_out_kernel<<<blocks, threads, 0, stream>>>(
            d_spectrum, output + b * freq_bins * frames, freq_bins, frames
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_padded));
    CUDA_CHECK(cudaFree(d_windowed_frames));
    CUDA_CHECK(cudaFree(d_spectrum));
    CUFFT_CHECK(cufftDestroy(plan));
}

// Kernels used by STFT
__global__ void stft_frames_kernel(
    const float* __restrict__ padded,   // padded_len
    float* __restrict__ windowed_frames, // frames * n_fft
    int frames,
    int n_fft,
    int hop_length,
    int win_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = frames * n_fft;
    if (idx < total) {
        int frame = idx / n_fft;
        int sample = idx % n_fft;
        int padded_idx = frame * hop_length + sample;
        float val = padded[padded_idx];
        // Hann window of length win_length, centered in n_fft as in torch.stft default.
        // torch.stft uses center=True and pads window to n_fft if win_length < n_fft.
        // The window is centered, meaning zeros before/after.
        int offset = (n_fft - win_length) / 2;
        float window_val = 0.0f;
        if (sample >= offset && sample < offset + win_length) {
            int w_idx = sample - offset;
            // Hann window: 0.5 * (1 - cos(2*pi*w_idx/(win_length-1)))
            double phase = 2.0 * M_PI * w_idx / (win_length - 1);
            window_val = 0.5 * (1.0 - cos(phase));
        }
        windowed_frames[idx] = val * window_val;
    }
}

__global__ void stft_copy_out_kernel(
    const cuComplex* __restrict__ spectrum, // [frames, freq_bins]
    cuComplex* __restrict__ output,          // [freq_bins, frames]
    int freq_bins,
    int frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = freq_bins * frames;
    if (idx < total) {
        int f = idx % frames;
        int fb = idx / frames;
        // spectrum is [frames, freq_bins] row-major
        cuComplex val = spectrum[f * freq_bins + fb];
        output[fb * frames + f] = val;
    }
}

// -----------------------------------------------------------------------------
// Conv2d via cuDNN
// -----------------------------------------------------------------------------
extern "C" void conv2d_forward_launcher(
    const float* input,       // [batch, in_channels, in_height, in_width]
    const float* weight,      // [out_channels, in_channels, kernel_h, kernel_w]
    const float* bias,        // [out_channels] (can be null)
    float* output,            // [batch, out_channels, out_height, out_width]
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w,
    int dilation_h,
    int dilation_w,
    cudaStream_t stream
) {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));

    // Set input descriptor: NCHW
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, in_channels, in_height, in_width));
    // Set filter descriptor
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           out_channels, in_channels, kernel_h, kernel_w));
    // Set convolution descriptor
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding_h, padding_w,
                                                stride_h, stride_w,
                                                dilation_h, dilation_w,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Determine output dimensions
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch, out_channels, out_height, out_width));
    // Set bias descriptor: one bias per channel, shape [1, out_channels, 1, 1]
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           1, out_channels, 1, 1));

    // Choose algorithm
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn, input_desc, filter_desc, conv_desc,
                                                        output_desc, algo, &workspace_size));
    void* d_workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
    }

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
                                        &alpha,
                                        input_desc, input,
                                        filter_desc, weight,
                                        conv_desc, algo,
                                        d_workspace, workspace_size,
                                        &beta,
                                        output_desc, output));

    // Add bias if present
    if (bias != nullptr) {
        // Use cudnnAddTensor
        float alpha_bias = 1.0f, beta_bias = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(cudnn, &alpha_bias, bias_desc, bias, &beta_bias, output_desc, output));
    }

    if (d_workspace) CUDA_CHECK(cudaFree(d_workspace));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}
