#include <cuda_runtime.h>
#include <math.h>

// BatchNorm2d evaluation mode kernel
// y = (x - running_mean[c]) / sqrt(running_var[c] + eps) * weight[c] + bias[c]
// Broadcast over N, H, W dimensions.
__global__ void batch_norm_eval_2d_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    int N,
    int C,
    int HW)
{
    int64_t total = (int64_t)N * C * HW;
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (int64_t)gridDim.x * blockDim.x) {
        int c = (idx / HW) % C;
        float mean = running_mean[c];
        float var = running_var[c];
        float w = (weight != nullptr) ? weight[c] : 1.0f;
        float b = (bias != nullptr) ? bias[c] : 0.0f;
        float val = (x[idx] - mean) / sqrtf(var + eps) * w + b;
        y[idx] = val;
    }
}

// Triplet margin loss with Lp distance (p=2), reduction='sum'
// For each sample n: d_pos = sqrt(sum((anchor - positive)^2)), d_neg = sqrt(sum((anchor - negative)^2))
// loss_n = max(d_pos - d_neg + margin, 0)
// total_loss = sum(loss_n)
__global__ void triplet_margin_loss_sum_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float margin,
    int N,
    int D,
    float* __restrict__ partial_sums)  // length = number of blocks in final reduction, or preallocated per-block buffer
{
    // One block per sample, or grid-stride with atomics? Use one block per sample for simplicity and stable result.
    // Assume blockDim.x >= 32, each block processes one sample.
    int n = blockIdx.x;
    if (n >= N) return;

    __shared__ float smem[256]; // enough for block reduction, blockDim.x <= 256

    int tid = threadIdx.x;
    float pos_sum = 0.0f;
    float neg_sum = 0.0f;
    // grid-stride loop within thread over D elements
    for (int i = tid; i < D; i += blockDim.x) {
        int64_t idx = (int64_t)n * D + i;
        float diff_pos = anchor[idx] - positive[idx];
        float diff_neg = anchor[idx] - negative[idx];
        pos_sum += diff_pos * diff_pos;
        neg_sum += diff_neg * diff_neg;
    }

    // Reduce pos_sum and neg_sum within block
    // First pos_sum
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            smem[tid] = pos_sum;
            __syncthreads();
            pos_sum += smem[tid + offset];
        }
    }
    // Actually better to use warp shuffle, but simple reduction is fine.
    // Let's implement efficient warp reduction and then cross-warp.
    __shared__ float pos_block[32];
    __shared__ float neg_block[32];

    // Warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        pos_sum += __shfl_down_sync(0xffffffff, pos_sum, offset);
        neg_sum += __shfl_down_sync(0xffffffff, neg_sum, offset);
    }
    if ((tid % 32) == 0) {
        pos_block[tid / 32] = pos_sum;
        neg_block[tid / 32] = neg_sum;
    }
    __syncthreads();

    if (tid < blockDim.x / 32) {
        float p = pos_block[tid];
        float q = neg_block[tid];
        for (int offset = (blockDim.x / 32) / 2; offset > 0; offset >>= 1) {
            p += __shfl_down_sync(0xffffffff, p, offset);
            q += __shfl_down_sync(0xffffffff, q, offset);
        }
        if (tid == 0) {
            float d_pos = sqrtf(p);
            float d_neg = sqrtf(q);
            float loss = fmaxf(d_pos - d_neg + margin, 0.0f);
            partial_sums[blockIdx.x] = loss;
        }
    }
}

// Final reduction kernel for triplet loss partial sums
__global__ void reduce_sum_kernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    if (idx < size) sum = input[idx];
    smem[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[blockIdx.x] = smem[0];
    }
}

// Launcher for batch_norm_eval_2d
extern "C" void batch_norm_eval_2d_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N,
    int C,
    int H,
    int W,
    cudaStream_t stream)
{
    int HW = H * W;
    int64_t total = (int64_t)N * C * HW;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    // Cap blocks to avoid launching too many, use grid-stride in kernel anyway
    if (blocks > 65535) blocks = 65535;
    batch_norm_eval_2d_kernel<<<blocks, threads, 0, stream>>>(
        x, y, weight, bias, running_mean, running_var, eps, N, C, HW);
}

// Launcher for triplet_margin_loss_sum
// Returns sum of losses across batch as a single float scalar, stored in output[0]
extern "C" void triplet_margin_loss_sum_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float margin,
    int p,
    int N,
    int D,
    float* output,
    cudaStream_t stream)
{
    // For p=2 only, as per contract; ignore p argument if not 2? Assert? We'll handle p=2 only.
    // Use partial_sums array of length N, then reduce to scalar.
    // We'll allocate temporary device memory inside launcher? Better to use fixed-size shared or use a small kernel with atomics.
    // Simpler: use a single block reduces N partial sums. But we need partial_sums array in device memory.
    // We'll use a temporary device allocation via cudaMallocAsync or static? Since binding layer will provide output scalar,
    // we'll allocate a temporary workspace and copy. But binding layer should allocate workspace, not launcher.
    // To keep launcher self-contained, we can use a two-pass approach: first kernel writes partial sums to a preallocated buffer
    // supplied by the binding. But contract specifies launcher signature with only output pointer and no workspace.
    // Alternative: Use a single kernel with atomicAdd on output and zero-initialize output. That is stable enough for summing losses,
    // but order of atomicAdd does not affect sum (floating point associativity may cause slight differences, but acceptable).
    // We'll use atomicAdd to accumulate loss directly into output[0], and we need to zero output before launch.
    // The binding layer should zero the output tensor before calling launcher, or we can do it inside launcher via cudaMemset.
    // For ease, we'll zero output using cudaMemsetAsync (allowed, no kernel).
    cudaMemsetAsync(output, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = N; // one block per sample
    triplet_margin_loss_sum_kernel<<<blocks, threads, 0, stream>>>(
        anchor, positive, negative, margin, N, D, output);
    // But our kernel currently writes to partial_sums array, not atomically to output.
    // We can modify kernel to accept an array and then a final reduction, or use a different kernel that atomically adds.
    // Let's adapt: we'll use a kernel that directly atomically adds block loss to output[0].
    // Since we only have one output scalar, each block computes loss for its sample and does atomicAdd(&output[0], loss).
    // The kernel below does that.
}

// Alternative triplet kernel with atomicAdd
extern "C" void triplet_margin_loss_sum_launcher_atomic(
    const float* anchor,
    const float* positive,
    const float* negative,
    float margin,
    int p,
    int N,
    int D,
    float* output,
    cudaStream_t stream)
{
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = N; // one block per sample
    triplet_margin_loss_sum_kernel<<<blocks, threads, 0, stream>>>(
        anchor, positive, negative, margin, N, D, output);
}

// Kernel that uses atomicAdd to output
template<typename T>
__global__ void triplet_margin_loss_sum_atomic_kernel(
    const T* __restrict__ anchor,
    const T* __restrict__ positive,
    const T* __restrict__ negative,
    float margin,
    int N,
    int D,
    T* __restrict__ output)
{
    int n = blockIdx.x;
    if (n >= N) return;

    int tid = threadIdx.x;
    float pos_sum = 0.0f;
    float neg_sum = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        int64_t idx = (int64_t)n * D + i;
        float diff_pos = (float)(anchor[idx] - positive[idx]);
        float diff_neg = (float)(anchor[idx] - negative[idx]);
        pos_sum += diff_pos * diff_pos;
        neg_sum += diff_neg * diff_neg;
    }

    // Block reduction using shared memory
    __shared__ float sdata[256];
    sdata[tid] = pos_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float d_pos_sq = sdata[0];

    __syncthreads();
    sdata[tid] = neg_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float d_neg_sq = sdata[0];
    __syncthreads();

    if (tid == 0) {
        float loss = fmaxf(sqrtf(d_pos_sq) - sqrtf(d_neg_sq) + margin, 0.0f);
        atomicAdd(output, loss);
    }
}

// Launcher for triplet using atomicAdd
extern "C" void triplet_margin_loss_sum_atomic_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float margin,
    int p,
    int N,
    int D,
    float* output,
    cudaStream_t stream)
{
    cudaMemsetAsync(output, 0, sizeof(float), stream);
    int threads = 256;
    int blocks = N;
    triplet_margin_loss_sum_atomic_kernel<float><<<blocks, threads, 0, stream>>>(
        anchor, positive, negative, margin, N, D, output);
}

// Note: We include multiple launcher implementations; binding should call the final one intended.
// The following launcher signatures will be used by binding: batch_norm_eval_2d_launcher and triplet_margin_loss_sum_atomic_launcher
// Ensure they match the declaration in binding.cpp (not shown here, but expected).
