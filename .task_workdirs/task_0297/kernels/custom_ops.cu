#include <cuda_runtime.h>
#include <cfloat>
#include <climits>

// Helper: get linear index from 4D coordinates using strides
__device__ __forceinline__ long long get_index_4d(long long b, long long c, long long h, long long w,
                                                  long long stride_b, long long stride_c, long long stride_h, long long stride_w) {
    return b * stride_b + c * stride_c + h * stride_h + w * stride_w;
}

// Kernel for log_softmax along dimension 1 (channels) on 4D tensor [B, C, H, W]
// Each block handles one (b, h, w) position, with threads reducing over C.
// Block size is chosen as next power of two >= C, but capped at 1024.
// Shared memory holds partial max and sum.
__global__ void log_softmax_dim1_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long B, long long C, long long H, long long W,
    long long stride_b, long long stride_c, long long stride_h, long long stride_w)
{
    extern __shared__ float smem[]; // first half for max, second half for sum
    float* smem_max = smem;
    float* smem_sum = &smem[blockDim.x];

    long long idx = blockIdx.x; // one block per (b,h,w)
    long long total_blocks = B * H * W;
    long long b = idx / (H * W);
    long long rem = idx - b * (H * W);
    long long h = rem / W;
    long long w = rem - h * W;

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // Initialize shared
    smem_max[tid] = -FLT_MAX;
    smem_sum[tid] = 0.0f;
    __syncthreads();

    // Find max over C using grid-stride within block (if C > nthreads)
    for (long long c = tid; c < C; c += nthreads) {
        long long offset = get_index_4d(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
        float val = input[offset];
        if (val > smem_max[tid]) smem_max[tid] = val;
    }

    // Reduce max across threads in block using shared memory tree reduction
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            float other = smem_max[tid + s];
            if (other > smem_max[tid]) smem_max[tid] = other;
        }
    }
    __syncthreads();
    float max_val = smem_max[0];

    // Compute sum of exp(x - max) over C
    for (long long c = tid; c < C; c += nthreads) {
        long long offset = get_index_4d(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
        float val = input[offset];
        float exp_val = __expf(val - max_val);
        smem_sum[tid] += exp_val;
    }

    // Reduce sum
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        __syncthreads();
        if (tid < s) {
            smem_sum[tid] += smem_sum[tid + s];
        }
    }
    __syncthreads();
    float sum_val = smem_sum[0];
    // Avoid division by zero; sum should be >0 if max was finite, but handle edge
    float log_sum = (sum_val > 0.0f) ? logf(sum_val) : 0.0f;

    // Write output for all C
    for (long long c = tid; c < C; c += nthreads) {
        long long offset = get_index_4d(b, c, h, w, stride_b, stride_c, stride_h, stride_w);
        output[offset] = input[offset] - max_val - log_sum;
    }
}

// Kernel for triu on last two dimensions of 4D tensor [B, C, H, W]
// diagonal parameter is passed; default 0. Operation: zero if w - h < diagonal? Careful: torch.triu keeps elements where col - row >= diagonal.
// For matrix H x W (rows H, cols W), triu keeps element if col >= row + diagonal. So zero if col < row + diagonal.
__global__ void triu_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    long long total_elements,
    long long B, long long C, long long H, long long W,
    long long stride_b, long long stride_c, long long stride_h, long long stride_w,
    int diagonal)
{
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (long long i = idx; i < total_elements; i += stride) {
        // extract b,c,h,w from linear index i (assuming contiguous layout for total elements? But input may be non-contiguous, stride provided.
        // Need to map from linear index to 4D coordinates using the strides? Actually output tensor also has same strides, so we can directly use i as offset if both input and output contiguous? But contract says supports strides, so we must honor strides.
        // We'll compute coordinates using modulo arithmetic based on shape dimensions, then compute offset using strides.
        // For that we need to map i (0..numel-1) to coordinates assuming a contiguous logical view; then use strides to get memory offset.
        long long tmp = i;
        long long w = tmp % W;
        tmp /= W;
        long long h = tmp % H;
        tmp /= H;
        long long c = tmp % C;
        tmp /= C;
        long long b = tmp;
        // now check condition
        if (w - h < diagonal) {
            output[get_index_4d(b, c, h, w, stride_b, stride_c, stride_h, stride_w)] = 0.0f;
        } else {
            output[get_index_4d(b, c, h, w, stride_b, stride_c, stride_h, stride_w)] = 
                input[get_index_4d(b, c, h, w, stride_b, stride_c, stride_h, stride_w)];
        }
    }
}

// Launcher for log_softmax
extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    long long B, long long C, long long H, long long W,
    long long stride_b, long long stride_c, long long stride_h, long long stride_w,
    cudaStream_t stream)
{
    if (B == 0 || C == 0 || H == 0 || W == 0) return;
    long long total_positions = B * H * W;
    // Choose block size: next power of two >= C, capped at 1024
    int block_size = 256;
    while (block_size < C && block_size < 1024) block_size <<= 1;
    if (block_size > 1024) block_size = 1024;
    // If C is very large, we may need to loop within thread, fine.
    dim3 grid((unsigned int)total_positions);
    size_t shared_mem = 2 * block_size * sizeof(float); // max and sum arrays
    log_softmax_dim1_kernel<<<grid, block_size, shared_mem, stream>>>(
        input, output, B, C, H, W, stride_b, stride_c, stride_h, stride_w);
}

// Launcher for triu
extern "C" void triu_forward_launcher(
    const float* input,
    float* output,
    long long B, long long C, long long H, long long W,
    long long stride_b, long long stride_c, long long stride_h, long long stride_w,
    int diagonal,
    cudaStream_t stream)
{
    long long total = B * C * H * W;
    if (total == 0) return;
    int block_size = 256;
    int grid_size = (int)min((total + block_size - 1) / block_size, 65535LL);
    triu_kernel<<<grid_size, block_size, 0, stream>>>(
        input, output, total, B, C, H, W, stride_b, stride_c, stride_h, stride_w, diagonal);
}
