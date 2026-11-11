#include <cuda_runtime.h>

// Correct rot90_clone_forward: out[b,w,c,h] = in[b,c,H-1-h,w]
// torch.rot90(x, k=1, dims=(1,2)) = transpose(dims 1,2) then flip on new dim2
__global__ void rot90_clone_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int batch,
                                   int channels,
                                   int height,
                                   int width) {
    int64_t total = (int64_t)batch * width * channels * height;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        // Decompose output linear index: i = ((b * W + w) * C + c) * H + h
        int h = (int)(i % height);
        int64_t tmp = i / height;
        int c = (int)(tmp % channels);
        tmp /= channels;
        int w = (int)(tmp % width);
        int b = (int)(tmp / width);

        // Input index: ((b * C + c) * H + (H-1-h)) * W + w
        int64_t in_idx = (((int64_t)b * channels + c) * height + (height - 1 - h)) * width + w;
        output[i] = input[in_idx];
    }
}

extern "C" void rot90_clone_launcher(const float* input,
                                       float* output,
                                       int batch,
                                       int channels,
                                       int height,
                                       int width,
                                       cudaStream_t stream) {
    int64_t total = (int64_t)batch * width * channels * height;
    if (total <= 0) return;

    int block_size = 256;
    int sm_count = 0;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);
    int max_blocks = (int)((total + block_size - 1) / block_size);
    int grid_size = sm_count * 8;
    if (grid_size > max_blocks) grid_size = max_blocks;
    if (grid_size < 1) grid_size = 1;

    rot90_clone_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, batch, channels, height, width);
}
