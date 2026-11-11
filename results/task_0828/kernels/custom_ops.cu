// CUDA implementation of STFT (Short-Time Fourier Transform) matching torch.stft with default parameters:
// center=True, pad_mode='reflect', normalized=False, onesided=True, window=None (rectangular), return_complex=True.
// Input: float32 tensor shape [batch, samples], contiguous.
// Output: complex64 tensor shape [batch, n_freq, n_frames], with strides (n_freq*n_frames, 1, n_freq) as in reference.
// This file contains only CUDA kernels and launchers with raw pointers and scalars.

#include <cuda_runtime.h>
#include <cstdint>

#define STFT_BLOCK_SIZE 256
#define MAX_FFT_SIZE 4096

__device__ __forceinline__ int reflect_index(int idx, int length) {
    if (idx < 0) {
        return -idx;
    } else if (idx >= length) {
        return 2 * length - idx - 2;
    } else {
        return idx;
    }
}

__device__ __forceinline__ void complex_mul(float a, float b, float c, float d, float &re, float &im) {
    re = a * c - b * d;
    im = a * d + b * c;
}

__device__ __forceinline__ void twiddle(int k, int N, float &re, float &im) {
    const float PI = 3.14159265358979323846f;
    float theta = -2.0f * PI * k / N;
    re = cosf(theta);
    im = sinf(theta);
}

__global__ void stft_kernel(
    const float* __restrict__ input,
    float2* __restrict__ output,
    int batch,
    int samples,
    int n_fft,
    int hop_length,
    int num_frames,
    long long input_stride_batch,
    long long output_batch_stride,
    int n_freq
) {
    int frame_idx = blockIdx.x;
    if (frame_idx >= batch * num_frames) return;
    int b = frame_idx / num_frames;
    int t = frame_idx % num_frames;

    int start = t * hop_length - n_fft / 2;

    __shared__ float s_real[MAX_FFT_SIZE];
    __shared__ float s_imag[MAX_FFT_SIZE];

    for (int i = threadIdx.x; i < n_fft; i += blockDim.x) {
        int pos = start + i;
        int idx = reflect_index(pos, samples);
        s_real[i] = input[b * input_stride_batch + idx];
        s_imag[i] = 0.0f;
    }
    __syncthreads();

    int log2_n = 0;
    int temp = n_fft;
    while (temp > 1) { temp >>= 1; log2_n++; }

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

    for (int stage = 1; stage <= log2_n; stage++) {
        int m = 1 << stage;
        int half_m = m >> 1;
        for (int k = threadIdx.x; k < half_m; k += blockDim.x) {
            float w_re, w_im;
            twiddle(k, m, w_re, w_im);
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

    // Write output in [batch, n_freq, num_frames] contiguous layout:
    // out_idx = b * output_batch_stride + f * num_frames + t
    for (int f = threadIdx.x; f < n_freq; f += blockDim.x) {
        long long out_idx = (long long)b * output_batch_stride + (long long)f * num_frames + t;
        output[out_idx] = make_float2(s_real[f], s_imag[f]);
    }
}

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
    int n_freq = n_fft / 2 + 1;
    long long output_batch_stride = (long long)n_freq * num_frames;
    long long input_batch_stride = samples;

    if (n_fft > MAX_FFT_SIZE) {
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
