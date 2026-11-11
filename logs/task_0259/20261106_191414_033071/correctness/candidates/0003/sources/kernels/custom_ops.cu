#include <cuda_runtime.h>
#include <cstdint>

__global__ void stft_kernel(const float* __restrict__ input,
                            float2* __restrict__ output,
                            int batch,
                            int signal_length,
                            int n_fft,
                            int hop_length,
                            int n_freq,
                            int n_frames,
                            float* __restrict__ window) {
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
        if (padded_pos < 0) padded_pos = 0;
        if (padded_pos >= signal_length) padded_pos = signal_length - 1;

        float w = window[i];
        float sample = signal[padded_pos] * w;
        float angle = -2.0f * 3.14159265358979323846f * f * i / n_fft;
        result.x += sample * cosf(angle);
        result.y += sample * sinf(angle);
    }

    // Write with stride [n_freq*n_frames, 1, n_freq]
    int output_offset = b * (n_freq * n_frames) + t * n_freq + f;
    output[output_offset] = result;
}

extern "C" void stft_forward_launcher(
    const float* input,
    float2* output,
    int batch,
    int signal_length,
    int n_fft,
    int hop_length,
    cudaStream_t stream) {

    int n_freq = n_fft / 2 + 1;
    int n_frames = 1 + signal_length / hop_length;

    int total = batch * n_freq * n_frames;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    float* window_dev = nullptr;
    cudaMalloc(&window_dev, n_fft * sizeof(float));
    float* window_host = (float*)malloc(n_fft * sizeof(float));
    for (int i = 0; i < n_fft; ++i) {
        window_host[i] = 0.5f - 0.5f * cosf(2.0f * 3.14159265358979323846f * i / n_fft);
    }
    cudaMemcpy(window_dev, window_host, n_fft * sizeof(float), cudaMemcpyHostToDevice);
    free(window_host);

    stft_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, signal_length, n_fft, hop_length, n_freq, n_frames, window_dev);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // error handling omitted
    }
    cudaFree(window_dev);
}

__global__ void reshape_copy_kernel(const float2* __restrict__ input,
                                    float2* __restrict__ output,
                                    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        output[idx] = input[idx];
    }
}

extern "C" void reshape_forward_launcher(
    const float2* input,
    float2* output,
    int total,
    cudaStream_t stream) {
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    reshape_copy_kernel<<<blocks, threads, 0, stream>>>(input, output, total);
}
