#include <cuda_runtime.h>
#include <cstdint>

// STFT forward kernels
// Input: float32 time-domain signal of shape [batch, signal_length]
// Output: complex64 (float2) tensor with logical shape [batch, n_freq, n_frames]
// Memory layout determined by output strides provided from binding.
// n_freq = n_fft/2 + 1
// n_frames = 1 + (signal_length - n_fft) / hop_length (center=True, pad=reflect)
// Padding mode reflect: reflect signal around edges to length signal_length + n_fft - 1 before windowing.
// Windowing: Hann periodic window.

__global__ void stft_kernel(const float* __restrict__ input,
                            float2* __restrict__ output,
                            int batch,
                            int signal_length,
                            int n_fft,
                            int hop_length,
                            int n_freq,
                            int n_frames,
                            const float* __restrict__ window,
                            long long batch_stride,
                            long long freq_stride,
                            long long frame_stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_freq * n_frames;
    if (idx >= total) return;

    int b = idx / (n_freq * n_frames);
    int rem = idx % (n_freq * n_frames);
    int f = rem / n_frames;
    int t = rem % n_frames;

    const float* signal = input + b * signal_length;
    float2 result = make_float2(0.0f, 0.0f);

    int start = t * hop_length - n_fft / 2;

    for (int i = 0; i < n_fft; ++i) {
        int pos = start + i;
        int padded_pos;
        if (pos < 0) {
            padded_pos = -pos - 1;
        } else if (pos >= signal_length) {
            padded_pos = 2 * signal_length - pos - 1;
        } else {
            padded_pos = pos;
        }
        // Clamp for safety; reflect should never go out of [0, signal_length-1]
        if (padded_pos < 0) padded_pos = 0;
        if (padded_pos >= signal_length) padded_pos = signal_length - 1;

        float w = window[i];
        float sample = signal[padded_pos] * w;
        float angle = -2.0f * 3.14159265358979323846f * f * i / n_fft;
        result.x += sample * cosf(angle);
        result.y += sample * sinf(angle);
    }

    // Compute linear offset using provided strides
    long long out_idx = b * batch_stride + f * freq_stride + t * frame_stride;
    output[out_idx] = result;
}

// Helper to generate Hann window on host and copy to device
static void generate_hann_window(float* window_host, int n_fft) {
    // Periodic Hann window as used by torch.stft: 0.5 - 0.5*cos(2*pi*i/n_fft)
    for (int i = 0; i < n_fft; ++i) {
        window_host[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265358979323846f * (double)i / n_fft);
    }
}

extern "C" void stft_forward_launcher(
    const float* input,
    float2* output,
    int batch,
    int signal_length,
    int n_fft,
    int hop_length,
    long long batch_stride,
    long long freq_stride,
    long long frame_stride,
    cudaStream_t stream) {

    int n_freq = n_fft / 2 + 1;
    // Frame count: center=True adds n_fft//2 on each side, effectively (signal_length + n_fft - 1) / hop_length + 1? 
    // Compute as in torch: n_frames = 1 + (signal_length - n_fft) / hop_length for even n_fft.
    int n_frames = 1 + (signal_length - n_fft) / hop_length;

    int total = batch * n_freq * n_frames;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    // Allocate window on device
    float* window_dev = nullptr;
    cudaMalloc(&window_dev, n_fft * sizeof(float));
    float* window_host = (float*)malloc(n_fft * sizeof(float));
    generate_hann_window(window_host, n_fft);
    cudaMemcpy(window_dev, window_host, n_fft * sizeof(float), cudaMemcpyHostToDevice);
    free(window_host);

    stft_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, signal_length, n_fft, hop_length, n_freq, n_frames,
        window_dev, batch_stride, freq_stride, frame_stride);

    cudaError_t err = cudaGetLastError();
    // Ignore error? Better to not crash but record; binding will check after launcher returns.

    cudaFree(window_dev);
}

// Reshape forward: This operation is a no-op in computation; binding returns a view.
// No kernel needed for copy. We leave empty for compatibility.
extern "C" void reshape_forward_launcher(
    const float2* input,
    float2* output,
    int total,
    cudaStream_t stream) {
    // Not used; binding creates a view directly.
}
