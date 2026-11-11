// CUDA implementation of STFT (Short-Time Fourier Transform) matching torch.stft with default parameters:
// center=True, pad_mode='reflect', normalized=False, onesided=True, window=None (rectangular), return_complex=True.
// Input: float32 tensor shape [batch, samples], contiguous.
// Output: complex64 tensor shape [batch, n_freq, n_frames], with strides (n_freq*n_frames, 1, n_freq) as in reference.
// This file contains only CUDA kernels and launchers with raw pointers and scalars.

#include <cuda_runtime.h>
#include <cstdint>

// Configuration: block size for FFT computation.
#define STFT_BLOCK_SIZE 256
#define MAX_FFT_SIZE 4096

// Device function for reflection padding index mapping.
// If idx is out of [0, length-1], return reflected index.
__device__ __forceinline__ int reflect_index(int idx, int length) {
    if (idx < 0) {
        // Reflect: e.g., -1 -> 0? PyTorch reflect pad: for pad left n, index -k maps to k? Let's verify.
        // For reflect mode, the padding is done by reflecting the input excluding the edge value.
        // For index -1: original index 0, reflected index should be 0? Actually torch.nn.functional.pad with mode='reflect' maps: out[i] = input[reflect_idx], where reflect_idx = -i-1? Let's test with small example.
        // PyTorch reflect pad: For 1D, pad=(left,right), reflect maps indices: -1 -> 1, -2 -> 2, etc. So idx < 0: return -idx.
        // For idx >= length: idx maps to 2*length - idx - 2? For length=4, idx=4 -> 2, idx=5 -> 1.
        // We'll implement based on known PyTorch behavior: for reflect mode, asymmetry: for left pad, reflect without repeating edge: index -1 becomes 1, -2 becomes 2; for right pad, index n becomes n-2, n+1 becomes n-3, etc.
        // So: if idx < 0: return -idx; else if idx >= length: return 2*length - idx - 2;
        // But need to ensure single reflection only; input signal length is large compared to n_fft, so one reflection sufficient.
        return -idx;
    } else if (idx >= length) {
        return 2 * length - idx - 2;
    } else {
        return idx;
    }
}

// Device function for complex multiply: (a+ib)*(c+id) = (ac-bd) + i(ad+bc)
__device__ __forceinline__ void complex_mul(float a, float b, float c, float d, float &re, float &im) {
    re = a * c - b * d;
    im = a * d + b * c;
}

// Device function for twiddle factor: exp(-2*pi*i*k/N) = cos(theta) - i*sin(theta)
__device__ __forceinline__ void twiddle(int k, int N, float &re, float &im) {
    const float PI = 3.14159265358979323846f;
    float theta = -2.0f * PI * k / N;
    re = cosf(theta);
    im = sinf(theta);
}

// Kernel: perform STFT for each frame independently.
// gridDim.x = num_frames_per_batch? We launch one block per (batch, frame) pair, so gridDim.x = batch * num_frames.
// Each block processes one frame of length n_fft (real input) and computes onesided FFT (n_fft/2+1 complex bins).
// Shared memory: real part of frame, and later reused for complex FFT stages.
__global__ void stft_kernel(
    const float* __restrict__ input,  // shape [batch, samples], contiguous
    float2* __restrict__ output,      // shape [batch, n_freq, n_frames], with strides batch_stride, freq_stride=1, time_stride=n_freq
    int batch,
    int samples,
    int n_fft,
    int hop_length,
    int num_frames,
    long long input_stride_batch,  // = samples (since contiguous)
    long long output_batch_stride, // = n_freq * num_frames (number of complex elements per batch)
    int n_freq                       // = n_fft/2 + 1
) {
    // frame index: blockIdx.x
    int frame_idx = blockIdx.x;
    int b = frame_idx / num_frames;
    int t = frame_idx % num_frames;

    // Start position of window in original signal after center padding: t*hop_length - n_fft/2
    int start = t * hop_length - n_fft / 2;

    // Load real signal into shared memory, apply reflection and window (rectangular, all ones).
    __shared__ float s_real[MAX_FFT_SIZE];
    __shared__ float s_imag[MAX_FFT_SIZE];

    // Initialize imaginary part to zero.
    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        int pos = start + i;
        int idx = reflect_index(pos, samples);
        s_real[i] = input[b * input_stride_batch + idx];
        s_imag[i] = 0.0f;
    }
    __syncthreads();

    // Perform complex FFT of length n_fft on s_real + i*s_imag, using iterative Cooley-Tukey radix-2.
    // Bit-reversal permutation first.
    int log2_n = 0;
    int temp = n_fft;
    while (temp > 1) { temp >>= 1; log2_n++; }

    // Bit reversal: swap elements.
    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        int rev = 0;
        int val = i;
        for (int bit = 0; bit < log2_n; bit++) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        if (i < rev) {
            float tr = s_real[i];
            float ti = s_imag[i];
            s_real[i] = s_real[rev];
            s_imag[i] = s_imag[rev];
            s_real[rev] = tr;
            s_imag[rev] = ti;
        }
    }
    __syncthreads();

    // FFT stages.
    for (int stage = 1; stage <= log2_n; stage++) {
        int m = 1 << stage;
        int half_m = m >> 1;
        // Twiddle factor for each k in [0, half_m-1]
        for (int k = threadIdx.x; k < half_m; k += blockDim.x) {
            float w_re, w_im;
            twiddle(k, m, w_re, w_im);
            // For each butterfly: for each group of size m, combine elements k and k+half_m
            for (int j = k; j < n_fft; j += m) {
                int idx1 = j;
                int idx2 = j + half_m;
                float u_re = s_real[idx1];
                float u_im = s_imag[idx1];
                float v_re = s_real[idx2];
                float v_im = s_imag[idx2];
                float t_re, t_im;
                complex_mul(v_re, v_im, w_re, w_im, t_re, t_im);
                s_real[idx1] = u_re + t_re;
                s_imag[idx1] = u_im + t_im;
                s_real[idx2] = u_re - t_re;
                s_imag[idx2] = u_im - t_im;
            }
        }
        __syncthreads();
    }

    // After FFT, we have full complex spectrum. Need onesided: only first n_freq bins.
    // Write to output, with layout: output[b, f, t] = output[(b * output_batch_stride) + f * output_freq_stride + t * output_time_stride]
    // freq_stride = 1, time_stride = n_freq.
    // output is float2 array, so complex value stored as float2 (real, imag).
    // For f in [0, n_freq-1], write s_real[f], s_imag[f].
    // Note: for real input, bins beyond n_fft/2 are redundant.
    for (int f = threadIdx.x; f < n_freq; f += blockDim.x) {
        // Compute linear index into output complex array: batch offset + frame*time_stride + freq*freq_stride
        // Since freq_stride=1, time_stride=n_freq, batch_stride= n_freq*num_frames
        long long out_idx = (long long)b * output_batch_stride + (long long)t * n_freq + f;
        output[out_idx] = make_float2(s_real[f], s_imag[f]);
    }
}

// Host launcher for STFT. Accepts raw pointers and scalars; caller (binding) ensures CUDA context and stream.
extern "C" void custom_stft_launcher(
    const float* input_ptr,
    float2* output_ptr,
    int batch,
    int samples,
    int n_fft,
    int hop_length,
    int num_frames,
    cudaStream_t stream
) {
    // Compute derived sizes.
    int n_freq = n_fft / 2 + 1;
    long long output_batch_stride = (long long)n_freq * num_frames;
    long long input_batch_stride = samples; // input is contiguous

    // Ensure n_fft <= MAX_FFT_SIZE
    if (n_fft > MAX_FFT_SIZE) {
        // This path should not be hit with given contract, but guard.
        return;
    }

    int total_frames = batch * num_frames;
    dim3 grid(total_frames);
    dim3 block(STFT_BLOCK_SIZE);

    stft_kernel<<<grid, block, 0, stream>>>(
        input_ptr,
        output_ptr,
        batch,
        samples,
        n_fft,
        hop_length,
        num_frames,
        input_batch_stride,
        output_batch_stride,
        n_freq
    );
}
