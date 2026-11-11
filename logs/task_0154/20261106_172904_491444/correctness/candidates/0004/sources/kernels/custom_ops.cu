// kernels/custom_ops.cu
// CUDA kernels and launchers for custom ops, no torch dependencies.
// Conv2d uses custom CUDA kernel (cuDNN unavailable); other ops use cuFFT.

#include <cuda_runtime.h>
#include <cufft.h>
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
// Output: float2* output: [batch, freq_bins, frames] (complex64 represented as float2)
// The launcher uses cuFFT for each frame.
// -----------------------------------------------------------------------------
extern "C" void stft_forward_launcher(
    const float* input,
    float2* output,
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
// Input: const float2* input: [batch, freq_bins, frames]
// Output: float* output: [batch, 2, freq_bins, frames]
// -----------------------------------------------------------------------------
__global__ void view_as_real_permute_kernel(
    const float2* __restrict__ input,
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

        float2 val = input[b * freq_bins * frames + fb * frames + f];
        // output layout: [batch, 2, freq_bins, frames]
        // real part at [b,0,fb,f], imag at [b,1,fb,f]
        output[b * 2 * freq_bins * frames + 0 * freq_bins * frames + fb * frames + f] = val.x;
        output[b * 2 * freq_bins * frames + 1 * freq_bins * frames + fb * frames + f] = val.y;
    }
}

extern "C" void view_as_real_permute_launcher(
    const float2* input,
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
    const float2* __restrict__ spectrum, // [frames, freq_bins]
    float2* __restrict__ output,          // [freq_bins, frames]
    int freq_bins,
    int frames
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = freq_bins * frames;
    if (idx < total) {
        int f = idx % frames;
        int fb = idx / frames;
        // spectrum is [frames, freq_bins] row-major
        float2 val = spectrum[f * freq_bins + fb];
        output[fb * frames + f] = val;
    }
}

// STFT launcher
void stft_forward_launcher(
    const float* input,
    float2* output,
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

    // Temporary device buffers
    float *d_padded, *d_windowed_frames;
    float2 *d_spectrum;

    CUDA_CHECK(cudaMalloc(&d_padded, padded_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_windowed_frames, frames * n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, frames * freq_bins * sizeof(float2)));

    // Create cuFFT plan for batched R2C
    cufftHandle plan;
    CUFFT_CHECK(cufftCreate(&plan));
    CUFFT_CHECK(cufftSetStream(plan, stream));
    // For batch of frames, use advanced plan with batch = frames, but we need to set n[0]=n_fft, inembed/onembed etc.
    // Simpler: create plan for single transform and call repeatedly? cufftPlanMany may be more efficient.
    // We'll use cufftPlanMany for batched transforms with stride 1, dist n_fft for input, dist freq_bins for output.
    int rank = 1;
    int n[] = {n_fft};
    int inembed[] = {0};
    int istride = 1, idist = n_fft;
    int onembed[] = {0};
    int ostride = 1, odist = freq_bins;
    CUFFT_CHECK(cufftPlanMany(&plan, rank, n,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_R2C, frames));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    int threads = 256;
    for (int b = 0; b < batch; ++b) {
        const float* input_b = input + b * sequence_length;

        // 1. Pad with reflection
        int blocks_pad = (padded_len + threads - 1) / threads;
        stft_pad_kernel<<<blocks_pad, threads, 0, stream>>>(
            input_b, d_padded, sequence_length, pad_left, pad_right, n_fft
        );
        CUDA_CHECK(cudaGetLastError());

        // 2. Extract frames and apply window
        int total_frames_samples = frames * n_fft;
        int blocks_frames = (total_frames_samples + threads - 1) / threads;
        stft_frames_kernel<<<blocks_frames, threads, 0, stream>>>(
            d_padded, d_windowed_frames, frames, n_fft, hop_length, win_length
        );
        CUDA_CHECK(cudaGetLastError());

        // 3. Batched cuFFT R2C on frames
        CUFFT_CHECK(cufftExecR2C(plan, d_windowed_frames, d_spectrum));

        // 4. Copy spectrum (freq_bins per frame) to output slice [b, freq_bins, frames]
        int total_out = freq_bins * frames;
        int blocks_out = (total_out + threads - 1) / threads;
        stft_copy_out_kernel<<<blocks_out, threads, 0, stream>>>(
            d_spectrum, output + b * freq_bins * frames, freq_bins, frames
        );
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaFree(d_padded));
    CUDA_CHECK(cudaFree(d_windowed_frames));
    CUDA_CHECK(cudaFree(d_spectrum));
    CUFFT_CHECK(cufftDestroy(plan));
}

// -----------------------------------------------------------------------------
// Conv2d via custom CUDA kernel (no cuDNN)
// -----------------------------------------------------------------------------
__global__ void conv2d_naive_kernel(
    const float* __restrict__ input,    // [batch, in_channels, in_height, in_width]
    const float* __restrict__ weight,   // [out_channels, in_channels, kernel_h, kernel_w]
    const float* __restrict__ bias,     // [out_channels], may be null
    float* __restrict__ output,         // [batch, out_channels, out_height, out_width]
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
    int out_height,
    int out_width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * out_channels * out_height * out_width;
    if (idx < total) {
        int ow = idx % out_width;
        int oh = (idx / out_width) % out_height;
        int oc = (idx / (out_width * out_height)) % out_channels;
        int b = idx / (out_width * out_height * out_channels);

        float sum = 0.0f;
        for (int ic = 0; ic < in_channels; ++ic) {
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int ih = oh * stride_h + kh * dilation_h - padding_h;
                    int iw = ow * stride_w + kw * dilation_w - padding_w;
                    if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                        float ival = input[((b * in_channels + ic) * in_height + ih) * in_width + iw];
                        float wval = weight[((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw];
                        sum += ival * wval;
                    }
                }
            }
        }
        if (bias != nullptr) {
            sum += bias[oc];
        }
        output[idx] = sum;
    }
}

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
    int out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    int total = batch * out_channels * out_height * out_width;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    conv2d_naive_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output,
        batch, in_channels, in_height, in_width,
        out_channels, kernel_h, kernel_w,
        stride_h, stride_w, padding_h, padding_w,
        dilation_h, dilation_w, out_height, out_width
    );
    CUDA_CHECK(cudaGetLastError());
}
