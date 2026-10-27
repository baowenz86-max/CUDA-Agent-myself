#include <cuda_runtime.h>
#include <cmath>

// -----------------------------------------------------------------------------
// adaptive_avg_pool2d_squeeze
// input: (B, C, H, W) float32, GPU pointer
// output: (B, C) float32, GPU pointer
// Computes mean over H,W for each b,c and writes output[b*C+c] = mean.
// Assumes input is contiguous (binding layer will make contiguous copy from CPU)
// -----------------------------------------------------------------------------
__global__ void adaptive_avg_pool2d_squeeze_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int H, int W) {
    int total = B * C;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int HW = H * W;
    for (int i = idx; i < total; i += stride) {
        int b = i / C;
        int c = i - b * C;
        const float* base = input + (size_t)b * C * HW + (size_t)c * HW;
        float sum = 0.0f;
        for (int j = 0; j < HW; ++j) {
            sum += base[j];
        }
        output[i] = sum / HW;
    }
}

extern "C" void adaptive_avg_pool2d_squeeze_launcher(
    const float* input_gpu,
    float* output_gpu,
    int B, int C, int H, int W,
    cudaStream_t stream) {
    int total = B * C;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // grid-stride loop handles arbitrary size
    adaptive_avg_pool2d_squeeze_kernel<<<blocks, threads, 0, stream>>>(
        input_gpu, output_gpu, B, C, H, W);
}

// -----------------------------------------------------------------------------
// triplet_margin_loss
// input: anchor, positive, negative of shape (B,C) float32 GPU pointers
// output: scalar float32 GPU pointer (single element)
// default margin=1.0, p=2, eps=1e-6, swap=False, reduction='mean'
// d_pos = sqrt(sum((a-p)^2, dim=1) + eps)
// d_neg = sqrt(sum((a-n)^2, dim=1) + eps)
// loss = max(d_pos - d_neg + margin, 0), mean over batch
// -----------------------------------------------------------------------------

// First stage: compute partial sums of squared differences for each item in batch.
// Each block handles one or more items? We'll use each thread for one item, but B=512
// so we can have one thread per item.
__global__ void triplet_partial_sums_kernel(
    const float* __restrict__ anchor,
    const float* __restrict__ positive,
    const float* __restrict__ negative,
    float* __restrict__ partial_pos,
    float* __restrict__ partial_neg,
    int B, int C) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const float* a = anchor + (size_t)b * C;
    const float* p = positive + (size_t)b * C;
    const float* n = negative + (size_t)b * C;
    float sum_pos = 0.0f;
    float sum_neg = 0.0f;
    for (int c = 0; c < C; ++c) {
        float diff_pos = a[c] - p[c];
        float diff_neg = a[c] - n[c];
        sum_pos += diff_pos * diff_pos;
        sum_neg += diff_neg * diff_neg;
    }
    partial_pos[b] = sum_pos;
    partial_neg[b] = sum_neg;
}

// Second stage: compute loss from partial sums, reduce mean across batch.
// Use one block with enough threads to cover B. Use shared memory reduction.
__global__ void triplet_final_reduce_kernel(
    const float* __restrict__ partial_pos,
    const float* __restrict__ partial_neg,
    float* __restrict__ output,
    int B,
    float margin,
    float eps) {
    __shared__ float s_loss[256]; // assume B <= 1024, use dynamic if needed but B=512 so 256 threads repeated
    int tid = threadIdx.x;
    int b = tid;
    float loss_val = 0.0f;
    if (b < B) {
        float d_pos = sqrtf(partial_pos[b] + eps);
        float d_neg = sqrtf(partial_neg[b] + eps);
        loss_val = fmaxf(d_pos - d_neg + margin, 0.0f);
    }
    s_loss[tid] = loss_val;
    __syncthreads();
    // reduction over blockDim.x threads; if B < blockDim.x, we need to consider only B elements.
    // Use standard tree reduction.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_loss[tid] += s_loss[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        // Divide by B for mean, but only elements contributing are tid<B, we initialized others to 0.
        output[0] = s_loss[0] / B;
    }
}

extern "C" void triplet_margin_loss_launcher(
    const float* anchor,
    const float* positive,
    const float* negative,
    float* output,
    int B, int C,
    cudaStream_t stream) {
    // Allocate temporary storage for partial sums (could be passed in, but we allocate here?
    // Binding layer can allocate, but spec says .cu launcher should not include torch; we can use cudaMalloc.
    // For simplicity, assume binding layer provides temporary buffers, but we'll do cudaMalloc inside launcher
    // to keep self-contained. However, that may cause overhead. Since B=512, small allocation fine.
    float *d_partial_pos, *d_partial_neg;
    cudaMalloc(&d_partial_pos, B * sizeof(float));
    cudaMalloc(&d_partial_neg, B * sizeof(float));
    // Launch partial sums kernel: one thread per item
    int threads_partial = 128;
    int blocks_partial = (B + threads_partial - 1) / threads_partial;
    triplet_partial_sums_kernel<<<blocks_partial, threads_partial, 0, stream>>>(
        anchor, positive, negative, d_partial_pos, d_partial_neg, B, C);
    // Launch final reduce kernel with enough threads to cover B, but max 1024, B=512 so 512 threads.
    int threads_final = 512; // B=512, fits
    // If B > 1024, need multiple blocks; but for this contract B=512.
    // We'll use one block of min(1024, next_pow2(B)) threads.
    int threads_reduce = 512; // since B=512
    triplet_final_reduce_kernel<<<1, threads_reduce, 0, stream>>>(
        d_partial_pos, d_partial_neg, output, B, 1.0f, 1e-6f);
    // Free temporary buffers after kernel completion? We should not free before kernel finishes.
    // Since kernel is async, we need to schedule free after kernel completion, but cudaFree on stream
    // may not be safe immediately. However, for this simple case, we can use cudaFree after launch;
    // cudaFree will synchronize? Actually cudaFree may block until kernel finishes if memory is in use.
    // But better to let binding layer manage memory? The instruction says .cu launcher should use raw pointers,
    // not allocate torch tensors. cudaMalloc/cudaFree is allowed. To avoid premature free, we can
    // cudaFree after launch and rely on implicit synchronization? Not guaranteed. Better to use
    // cudaMallocAsync? Not available. We can use a small pool? For simplicity and since this is a single
    // kernel sequence, we can call cudaDeviceSynchronize() inside launcher, but that defeats async.
    // Instead, we can have binding layer allocate temp buffers and pass them, but not specified.
    // We'll allocate using cudaMalloc, then launch kernels, then cudaFree without synchronization.
    // cudaFree on stream-ordered memory returns immediately but actually defers until all preceding work done.
    // That's valid: cudaFree enqueues a free operation on the stream? Actually cudaFree is synchronous with
    // respect to device; it will block until all previously launched work completes, then free. That would
    // cause synchronization, which is acceptable for correctness but may hurt performance slightly.
    // Given the problem size and focus on correctness, we accept this. Alternatively, we could use
    // cudaMalloc and cudaFree with cudaStreamSynchronize, but same effect.
    cudaFree(d_partial_pos);
    cudaFree(d_partial_neg);
}
