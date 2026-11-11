#include <cuda_runtime.h>
#include <cuComplex.h>

// ============================================================================
// STFT CUDA kernels
// ============================================================================
// Reference: torch.stft(input, n_fft, hop_length, win_length, return_complex=True)
// Uses rectangular window (None), center=True, pad_mode='reflect', normalized=False, onesided=True.
// Input layout: [batch, num_samples], float32, contiguous (or at least stride[1]=1).
// Output layout: complex64 with shape [batch, n_fft//2 + 1, frames],
//    and strides as in torch.stft: [frames * (n_fft//2+1), 1, n_fft//2+1].
// frames = 1 + (num_samples - n_fft) / hop_length.
// For onesided, output frequency bins = n_fft//2 + 1.

// Utility to compute reflect padding index for a given index i in padded signal.
// Padded length = num_samples + 2 * pad, where pad = n_fft // 2 (center=True default).
__device__ inline int reflect_index(int i, int num_samples, int pad) {
    int N = num_samples;
    int total = N + 2 * pad;
    if (i < pad) {
        // reflect about boundary between pad and pad-1: index in original = pad - 1 - (i - pad) = 2*pad - 1 - i
        return 2 * pad - 1 - i;
    } else if (i >= pad + N) {
        // reflect about boundary at pad+N: index = 2*(pad+N)-1 - i = 2*pad + 2*N -1 - i
        return 2 * pad + 2 * N - 1 - i;
    } else {
        return i - pad;
    }
}

// Precompute twiddle factors for each n_fft length, reused across all frames/batches.
__global__ void precompute_twiddle_kernel(cuComplex* twiddle, int n_fft) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_fft) return;
    float angle = -2.0f * 3.14159265358979323846f * (float)k / (float)n_fft;
    twiddle[k] = make_cuComplex(cosf(angle), sinf(angle));
}

// STFT kernel: each thread computes one (batch, freq_bin, frame) output element.
// Input: float* input, shape [batch, num_samples], contiguous (but we use stride via num_samples).
// Output: cuComplex* output, shape [batch, bins, frames], with strides: out_stride_batch = frames * bins, out_stride_frame = bins, out_stride_freq = 1.
__global__ void stft_kernel(
    const float* __restrict__ input,
    cuComplex* __restrict__ output,
    const cuComplex* __restrict__ twiddle,
    int batch,
    int num_samples,
    int n_fft,
    int hop_length,
    int win_length,
    int bins,
    int frames,
    int pad
) {
    int total_output = batch * bins * frames;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int linear = idx; linear < total_output; linear += stride) {
        // Decode linear index to (b, freq, frame)
        int frame = linear % frames;
        int tmp = linear / frames;
        int freq = tmp % bins;
        int b = tmp / bins;
        
        // Compute DFT for this (b, freq, frame)
        // Window is rectangular: for n < win_length, weight=1; for n >= win_length, weight=0.
        // But win_length == n_fft in this task, so no truncation.
        // Use win_length for generality, but assume win_length <= n_fft; if shorter, zero beyond win_length.
        int start_sample = frame * hop_length;  // because center=True, actual signal segment starts at start_sample - pad in input coordinate, but we use reflect index.
        
        cuComplex sum = make_cuComplex(0.0f, 0.0f);
        int n_eff = min(win_length, n_fft);
        for (int n = 0; n < n_eff; ++n) {
            // Index in padded signal: start_sample + n (since padded signal has length num_samples+2*pad, and we pad reflect)
            int padded_idx = start_sample + n;
            int orig_idx = reflect_index(padded_idx, num_samples, pad);
            // orig_idx is guaranteed in [0, num_samples-1] due to reflect padding.
            float x = input[b * num_samples + orig_idx];
            // Twiddle factor: exp(-2*pi*i * freq * n / n_fft)
            int twiddle_idx = (freq * n) % n_fft;
            cuComplex tw = twiddle[twiddle_idx];
            // sum += x * tw
            sum.x += x * tw.x;
            sum.y += x * tw.y;
        }
        // normalized=False, so no scaling.
        int out_idx = (b * frames + frame) * bins + freq;  // matches strides [frames*bins, 1, bins] ? Actually output stride: batch stride = frames*bins, frame stride = bins, freq stride = 1.
        output[out_idx] = sum;
    }
}

// Launcher function matching expected C interface used by binding.
// Arguments:
//   input: float pointer, shape [batch, num_samples]
//   output: cuComplex pointer, shape [batch, bins, frames], allocated by binding
//   batch, num_samples, n_fft, hop_length, win_length: int64_t
//   stream: cudaStream_t
extern "C" void custom_stft_launcher(
    const float* input,
    cuComplex* output,
    int64_t batch,
    int64_t num_samples,
    int64_t n_fft,
    int64_t hop_length,
    int64_t win_length,
    cudaStream_t stream
) {
    int64_t bins = n_fft / 2 + 1;
    int64_t frames = 1 + (num_samples - n_fft) / hop_length;
    int64_t pad = n_fft / 2;

    // Allocate twiddle factors once on device (could be optimized with constant memory, but simple).
    // We allocate inside launcher, then free after kernel. Better to cache but for simplicity.
    cuComplex* twiddle_device = nullptr;
    cudaMalloc(&twiddle_device, n_fft * sizeof(cuComplex));
    // Precompute twiddle factors
    int threads_pre = 256;
    int blocks_pre = (int)((n_fft + threads_pre - 1) / threads_pre);
    precompute_twiddle_kernel<<<blocks_pre, threads_pre, 0, stream>>>(twiddle_device, (int)n_fft);

    // Launch STFT kernel
    int64_t total_output = batch * bins * frames;
    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap, but kernel uses grid-stride loop
    stft_kernel<<<blocks, threads, 0, stream>>>(
        input,
        output,
        twiddle_device,
        (int)batch,
        (int)num_samples,
        (int)n_fft,
        (int)hop_length,
        (int)win_length,
        (int)bins,
        (int)frames,
        (int)pad
    );

    // Free twiddle after kernel completes (need synchronize? but stream ordering ensures kernel done before free? Actually cudaFree after kernel launch in same stream is safe because free waits for kernel? No, cudaFree is synchronous? It may block until kernel completes? According to CUDA, cudaFree does implicit synchronization. So it's fine.)
    cudaFree(twiddle_device);
}
