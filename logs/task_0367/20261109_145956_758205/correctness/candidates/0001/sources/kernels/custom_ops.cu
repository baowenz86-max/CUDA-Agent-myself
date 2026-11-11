// kernels/custom_ops.cu
// CUDA kernel for AdaptiveAvgPool2d forward (float32 only)
// No PyTorch/ATen dependencies, only cuda_runtime.h
#include <cuda_runtime.h>

// Kernel: one block per output element
// Each block uses blockDim.x threads to sum all elements in the corresponding adaptive pooling window
// Input tensor can be non-contiguous; strides are used for indexing
// dimensions: N, C, H_in, W_in, H_out, W_out
// global work size: N*C*H_out*W_out blocks
// blockDim.x: number of threads per block (configurable, e.g., 256)
__global__ void adaptive_avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H_in, int W_in,
    int H_out, int W_out,
    long long in_stride_n,   // stride in elements for batch
    long long in_stride_c,   // stride for channels
    long long in_stride_h,   // stride for height
    long long in_stride_w,   // stride for width
    long long out_stride_n,  // output strides (usually contiguous)
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w
) {
    int tid = blockIdx.x;  // one block per output element
    int total_blocks = N * C * H_out * W_out;
    int idx = tid;
    // grid-stride loop if gridDim.x may be less than total_blocks
    for (; idx < total_blocks; idx += gridDim.x) {
        // decode idx -> (n, c, oh, ow)
        int ow = idx % W_out;
        int tmp = idx / W_out;
        int oh = tmp % H_out;
        tmp /= H_out;
        int c = tmp % C;
        int n = tmp / C;

        // compute pooling window boundaries
        int h_start = (int)floorf((float)oh * H_in / H_out);
        int h_end   = (int)ceilf((float)(oh + 1) * H_in / H_out);
        int w_start = (int)floorf((float)ow * W_in / W_out);
        int w_end   = (int)ceilf((float)(ow + 1) * W_in / W_out);

        int kh = h_end - h_start;
        int kw = w_end - w_start;
        int window_size = kh * kw;
        if (window_size == 0) {
            // Should never happen for valid inputs, but guard
            output[n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w] = 0.0f;
            continue;
        }

        // Each thread accumulates a subset of elements
        float sum = 0.0f;
        int thread_id = threadIdx.x;
        int num_threads = blockDim.x;
        for (int i = thread_id; i < window_size; i += num_threads) {
            int local_h = i / kw;
            int local_w = i % kw;
            int h = h_start + local_h;
            int w = w_start + local_w;
            // ensure within input bounds (should be)
            if (h < H_in && w < W_in) {
                long long input_idx = (long long)n * in_stride_n +
                                      (long long)c * in_stride_c +
                                      (long long)h * in_stride_h +
                                      (long long)w * in_stride_w;
                sum += input[input_idx];
            }
        }

        // Block reduction to get total sum
        // Use shared memory for reduction within block
        extern __shared__ float sdata[];
        sdata[thread_id] = sum;
        __syncthreads();

        // reduction tree
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (thread_id < s) {
                sdata[thread_id] += sdata[thread_id + s];
            }
            __syncthreads();
        }

        if (thread_id == 0) {
            float avg = sdata[0] / (float)window_size;
            output[n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w] = avg;
        }
    }
}

// Host launcher: no PyTorch dependencies, only raw pointers and scalars
// output_size is the target H_out and W_out (same value for both dims)
// This function is called by the Python binding wrapper.
extern "C" void custom_adaptive_avg_pool2d_launcher(
    const float* input,
    float* output,
    int batch_size,
    int channels,
    int input_height,
    int input_width,
    int output_height,   // typically output_size
    int output_width,    // typically output_size
    long long in_stride_n,
    long long in_stride_c,
    long long in_stride_h,
    long long in_stride_w,
    long long out_stride_n,
    long long out_stride_c,
    long long out_stride_h,
    long long out_stride_w,
    cudaStream_t stream
) {
    int total_blocks = batch_size * channels * output_height * output_width;
    // Use a reasonable number of blocks: one block per output element, but cap to avoid excessive blocks
    // However, for this small output (5x5) total_blocks = N*C*25, which is small, so just use total_blocks.
    // But we also have a grid-stride loop in kernel to handle cases where gridDim.x < total_blocks.
    int block_size = 256;  // can be tuned
    // Determine grid size: ideally total_blocks, but we can set exactly total_blocks since it's small.
    // For safety, limit to 65535 blocks per grid (maximum for old archs, but modern allows more).
    // We'll set grid = min(total_blocks, 65535) and rely on grid-stride loop.
    int grid_size = (total_blocks < 65535) ? total_blocks : 65535;
    if (grid_size == 0) {
        // no elements, just return
        return;
    }
    // shared memory size: block_size floats
    size_t shared_mem = block_size * sizeof(float);
    adaptive_avg_pool2d_kernel<<<grid_size, block_size, shared_mem, stream>>>(
        input, output,
        batch_size, channels, input_height, input_width,
        output_height, output_width,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w
    );
    // Note: error checking is done by the binding after launch
}
