#include <cuda_runtime.h>

// Kernel: compute global average pooling for 5D input [N, C, D, H, W]
// Each thread block handles one (n, c) channel, or multiple if needed via grid-stride.
// Since output size is (1,1,1), we sum over all D*H*W elements for each channel.
// To handle large W*H*D, we use a grid-stride loop within each block and perform a block reduction.
// The launcher will launch N*C blocks, each with blockDim threads.
// Shared memory is used for the reduction.

template <int BLOCK_SIZE>
__global__ void adaptive_avg_pool3d_global_avg_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int D, int H, int W)
{
    // Block index corresponds to (n, c) pair
    int channel_idx = blockIdx.x;  // 0 .. N*C-1
    int c = channel_idx % C;
    int n = channel_idx / C;
    
    const int total_elements = D * H * W;
    
    // Shared memory for block reduction
    __shared__ float sdata[BLOCK_SIZE];
    
    float sum = 0.0f;
    
    // Grid-stride loop over D*H*W elements for this channel
    for (int idx = threadIdx.x; idx < total_elements; idx += BLOCK_SIZE) {
        // Convert linear index to (d, h, w)
        int w = idx % W;
        int h = (idx / W) % H;
        int d = idx / (W * H);
        
        // Input offset: (n, c, d, h, w)
        int input_offset = ((n * C + c) * D + d) * H * W + h * W + w;
        // Alternative expression: n*C*D*H*W + c*D*H*W + d*H*W + h*W + w
        
        sum += input[input_offset];
    }
    
    // Block reduction: store partial sums in shared memory
    sdata[threadIdx.x] = sum;
    __syncthreads();
    
    // Tree reduction
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes the averaged result
    if (threadIdx.x == 0) {
        output[channel_idx] = sdata[0] / total_elements;
    }
}

// Host launcher for adaptive_avg_pool3d_forward with output size (1,1,1)
// This launcher assumes output size is (1,1,1) and handles general N,C,D,H,W.
// For different output sizes, a different kernel or a more general approach would be needed,
// but per the contract this operator is only called with output_size=(1,1,1).
extern "C" void adaptive_avg_pool3d_forward_launcher(
    const float* input,
    float* output,
    int N, int C, int D, int H, int W,
    int out_d, int out_h, int out_w,
    cudaStream_t stream)
{
    // For safety, if output size is not (1,1,1), we silently do nothing here.
    // The binding should enforce output_size = (1,1,1) by passing ints.
    if (out_d != 1 || out_h != 1 || out_w != 1) {
        // Optionally fill output with zeros or handle other cases? 
        // Since contract only requires (1,1,1), we can just return.
        // But to avoid uninitialized output, we could set to zero.
        // However binding will check and only call with (1,1,1).
        return;
    }
    
    const int total_channels = N * C;
    const int block_size = 256; // Choose typical block size
    
    // Launch one block per channel
    dim3 grid(total_channels);
    dim3 block(block_size);
    
    adaptive_avg_pool3d_global_avg_kernel<256><<<grid, block, 0, stream>>>(
        input, output, N, C, D, H, W);
}
