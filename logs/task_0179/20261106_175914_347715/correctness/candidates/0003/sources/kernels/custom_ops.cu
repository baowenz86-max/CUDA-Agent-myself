#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

// 错误检查宏
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(err));                                           \
            return;                                                                     \
        }                                                                               \
    } while (0)

#define CUFFT_CHECK(call)                                                               \
    do {                                                                                \
        cufftResult err = call;                                                         \
        if (err != CUFFT_SUCCESS) {                                                     \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, (int)err);\
            return;                                                                     \
        }                                                                               \
    } while (0)

// move_to_cuda: 简单的异步复制
extern "C" void move_to_cuda_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(output, input, numel * sizeof(float), cudaMemcpyHostToDevice, stream));
}

// 内核：stft_abs 中对单个 batch 的复数帧取模
__global__ void stft_abs_kernel(
    const cufftComplex* complex_data,
    float* output,
    int n_fft,
    int num_freq,
    int num_frames
) {
    int freq = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y * blockDim.y + threadIdx.y;
    if (freq < num_freq && frame < num_frames) {
        cufftComplex val = complex_data[freq * num_frames + frame];
        output[freq * num_frames + frame] = sqrtf(val.x * val.x + val.y * val.y);
    }
}

// 内核：初始化 Hann 窗
__global__ void init_hann_window(float* window, int win_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < win_length) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (float)(win_length - 1)));
    }
}

// 反射填充内核
__global__ void reflect_pad_fill(float* padded, const float* input, int length, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = length + 2 * pad;
    if (idx >= total) return;
    if (idx < pad) {
        padded[idx] = input[pad - idx];
    } else if (idx >= pad + length) {
        int right_idx = idx - pad - length;
        padded[idx] = input[length - 2 - right_idx];
    } else {
        padded[idx] = input[idx - pad];
    }
}

// 窗口复制并乘以 Hann 窗内核
__global__ void build_frames(float* frames, const float* padded, const float* window, int length_padded, int n_fft, int hop_length, int num_frames) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= num_frames) return;
    int start = frame * hop_length;
    for (int i = 0; i < n_fft; ++i) {
        frames[frame * n_fft + i] = padded[start + i] * window[i];
    }
}

// 取模内核
__global__ void complex_abs_kernel(const cufftComplex* complex_data, float* output, int num_freq, int num_frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_freq * num_frames;
    if (idx >= total) return;
    int freq = idx % num_freq;
    int frame = idx / num_freq;
    cufftComplex val = complex_data[idx];
    output[freq * num_frames + frame] = sqrtf(val.x * val.x + val.y * val.y);
}

// stft_abs_launcher 定义
extern "C" void stft_abs_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t length,
    int n_fft,
    int hop_length,
    int win_length,
    cudaStream_t stream
) {
    if (batch_size <= 0 || length <= 0 || n_fft <= 0 || hop_length <= 0 || win_length <= 0) {
        return;
    }

    int center_pad = n_fft / 2;
    int num_frames = 1 + (int)((length + 2 * center_pad - n_fft) / (int64_t)hop_length);
    int num_freq = n_fft / 2 + 1;

    float* d_window;
    CUDA_CHECK(cudaMalloc(&d_window, win_length * sizeof(float)));
    int block = 256;
    int grid = (win_length + block - 1) / block;
    init_hann_window<<<grid, block, 0, stream>>>(d_window, win_length);

    int padded_len = length + 2 * center_pad;
    float* d_padded = nullptr;
    float* d_frames = nullptr;
    cufftComplex* d_complex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_padded, padded_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_frames, num_frames * n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_complex, num_frames * num_freq * sizeof(cufftComplex)));

    cufftHandle plan;
    int rank = 1;
    int n[] = {n_fft};
    int inembed[] = {n_fft};
    int istride = 1;
    int idist = n_fft;
    int onembed[] = {num_freq};
    int ostride = 1;
    int odist = num_freq;
    CUFFT_CHECK(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, num_frames));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * length;
        float* batch_output = output + b * num_freq * num_frames;

        int padded_grid = (padded_len + block - 1) / block;
        reflect_pad_fill<<<padded_grid, block, 0, stream>>>(d_padded, batch_input, (int)length, center_pad);

        int frames_grid = (num_frames + block - 1) / block;
        build_frames<<<frames_grid, block, 0, stream>>>(d_frames, d_padded, d_window, padded_len, n_fft, hop_length, num_frames);

        CUFFT_CHECK(cufftExecR2C(plan, d_frames, d_complex));

        int abs_total = num_freq * num_frames;
        int abs_grid = (abs_total + block - 1) / block;
        complex_abs_kernel<<<abs_grid, block, 0, stream>>>(d_complex, batch_output, num_freq, num_frames);
    }

    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_window));
    CUDA_CHECK(cudaFree(d_padded));
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_complex));
}

// reflection_pad_2d 内核
__global__ void reflection_pad_2d_kernel(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    int64_t stride_b,
    int64_t stride_h,
    int64_t stride_w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * (height + pad_top + pad_bottom) * (width + pad_left + pad_right);
    if (idx >= total) return;

    int out_width = width + pad_left + pad_right;
    int out_height = height + pad_top + pad_bottom;

    int64_t tmp = idx;
    int ow = tmp % out_width;
    tmp /= out_width;
    int oh = tmp % out_height;
    tmp /= out_height;
    int b = tmp;

    int ih = oh - pad_top;
    int iw = ow - pad_left;

    if (ih < 0) ih = -ih - 1;
    if (ih >= height) ih = 2 * height - ih - 1;
    if (iw < 0) iw = -iw - 1;
    if (iw >= width) iw = 2 * width - iw - 1;

    const float* input_ptr = input + b * stride_b + ih * stride_h + iw * stride_w;
    output[idx] = *input_ptr;
}

// reflection_pad_2d_launcher 定义
extern "C" void reflection_pad_2d_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    const int64_t* input_strides,
    cudaStream_t stream
) {
    int64_t stride_b = input_strides[0];
    int64_t stride_h = input_strides[1];
    int64_t stride_w = input_strides[2];

    int out_height = height + pad_top + pad_bottom;
    int out_width = width + pad_left + pad_right;
    int64_t total = batch_size * out_height * out_width;

    int block = 256;
    int grid = (int)((total + block - 1) / block);
    reflection_pad_2d_kernel<<<grid, block, 0, stream>>>(
        output, input, batch_size, height, width,
        pad_left, pad_right, pad_top, pad_bottom,
        stride_b, stride_h, stride_w
    );
}
