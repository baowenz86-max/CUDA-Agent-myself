#include <cuda_runtime.h>

// Correct rot90_clone_forward: out[b,w,c,h] = in[b,c,H-1-h,w]
// We produce a contiguous base tensor of shape [batch, channels, width, height]
// Then PyTorch binding returns base.permute({0,2,1,3}) to get expected shape [batch,width,channels,height]
// with expected strides [307200,320,102400,1].
__global__ void rot90_clone_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int batch,
                                   int channels,
                                   int height,
                                   int width) {
    // output is base tensor [batch, channels, width, height]
    // So linear index i decomposes as: i = ((b*C + c)*W + w)*H + h
    int64_t total = (int64_t)batch * channels * width * height;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t i = idx; i < total; i += stride) {
        int h = (int)(i % height);
        int64_t tmp = i / height;
        int w = (int)(tmp % width);
        tmp /= width;
        int c = (int)(tmp % channels);
        int b = (int)(tmp / channels);

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
    int64_t total = (int64_t)batch * channels * width * height;
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
