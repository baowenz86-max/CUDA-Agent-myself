// Binary Cross Entropy with logits loss, mean reduction.
// Input logits: [N] or any shape, target: same shape (float32 with 0/1).
// loss = mean( max(logit,0) - logit*target + log(1+exp(-|logit|)) )
// Only float32, contiguous, CUDA.

#include <cuda_runtime.h>
#include <math.h>

__global__ void bce_loss_reduce_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ targets,
    float* __restrict__ partial_sums, // one per block
    int numel
) {
    extern __shared__ float sdata[]; // shared memory for block reduction
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int i = idx; i < numel; i += stride) {
        float logit = logits[i];
        float target = targets[i];
        float loss = fmaxf(logit, 0.0f) - logit * target + log1pf(expf(-fabsf(logit)));
        local_sum += loss;
    }

    // Block reduction
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

extern "C" float bce_with_logits_loss_launcher(
    const float* logits,
    const float* targets,
    int numel,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024; // limit blocks for grid-stride loop
    // Allocate temporary storage for partial sums
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    bce_loss_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        logits, targets, d_partial, numel);
    // Copy partial sums to host and sum
    float* h_partial = new float[blocks];
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_partial);
    float loss = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        loss += h_partial[i];
    }
    delete[] h_partial;
    return loss / (float)numel;
}
