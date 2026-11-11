#include <cuda_runtime.h>

extern "C" {

// PixelShuffle kernel: rearrange input tensor (B, C*r^2, H, W) to output (B, C, H*r, W*r)
// y[b, c, h*r + i, w*r + j] = x[b, c*r^2 + i*r + j, h, w]
__global__ void pixel_shuffle_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int batch,
    int channels,        // C = output channels
    int height,          // H
    int width,           // W
    int upscale_factor,  // r
    long long input_batch_stride,
    long long input_channel_stride,
    long long input_height_stride,
    long long input_width_stride)
{
    long long total_output = (long long)batch * channels * height * upscale_factor * width * upscale_factor;
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total_output; idx += stride) {
        // Decode output linear index into (b, c, oh, ow)
        int ow = idx % (width * upscale_factor);
        long long tmp = idx / (width * upscale_factor);
        int oh = tmp % (height * upscale_factor);
        tmp /= (height * upscale_factor);
        int c = tmp % channels;
        int b = tmp / channels;

        int h = oh / upscale_factor;
        int w = ow / upscale_factor;
        int i = oh % upscale_factor;
        int j = ow % upscale_factor;

        int input_channel = c * upscale_factor * upscale_factor + i * upscale_factor + j;

        long long input_offset = (long long)b * input_batch_stride
                               + (long long)input_channel * input_channel_stride
                               + (long long)h * input_height_stride
                               + (long long)w * input_width_stride;

        output[idx] = input[input_offset];
    }
}

void pixel_shuffle_forward_launcher(
    float* output,
    const float* input,
    int batch,
    int channels,
    int height,
    int width,
    int upscale_factor,
    const long long* input_strides, // array of 4 strides in elements, not bytes
    cudaStream_t stream)
{
    long long total_output = (long long)batch * channels * height * upscale_factor * width * upscale_factor;
    if (total_output <= 0) return;

    int threads = 256;
    int blocks = (int)((total_output + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid cap, kernel uses grid-stride loop

    pixel_shuffle_kernel<<<blocks, threads, 0, stream>>>(
        output,
        input,
        batch,
        channels,
        height,
        width,
        upscale_factor,
        input_strides[0],
        input_strides[1],
        input_strides[2],
        input_strides[3]
    );
}

} // extern "C"
