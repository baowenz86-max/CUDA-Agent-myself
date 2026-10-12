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
    // Allocate temporary storage for partial sums (we need to do this without PyTorch, so we rely on caller to provide buffer?)
    // Since we need to return a scalar, we can do a two-stage reduction. We'll allocate a device buffer using cudaMalloc.
    float* partial_sums;
    cudaMalloc(&partial_sums, blocks * sizeof(float));
    bce_loss_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(
        logits, targets, partial_sums, numel);
    // Second reduction: copy partial sums to host and sum, or do another kernel? Simpler: use cudaMemcpy to host and sum on CPU? But we are in a launcher, we can do cudaMemcpyAsync to host and then sum? However, we need to return a float that the binding can put into a tensor.
    // Since the launcher is called from Python binding, we can allocate a pinned host memory and sum after device-to-host copy. But the binding will need to know the result. Alternatively, we can have the binding allocate output tensor and pass it, but our current interface defined by model_new.py expects a scalar tensor return from the function. The binding will handle creating the tensor. So we need the launcher to return float value after synchronization.
    // We'll do: allocate a temporary device buffer of one float for final sum, then a second kernel to reduce partial sums, then copy to host and return.
    // To keep it simple, we can allocate a pageable host memory and call cudaMemcpy with cudaMemcpyDeviceToHost after reduction to one block. We'll do a second kernel to sum partial sums into one value.
    float* final_sum;
    cudaMalloc(&final_sum, sizeof(float));
    // Launch reduction kernel for partial sums
    int reduce_blocks = 1;
    int reduce_threads = 256;
    // We'll reuse the same reduction kernel with numel = blocks, input = partial_sums, output = final_sum
    bce_loss_reduce_kernel<<<reduce_blocks, reduce_threads, reduce_threads * sizeof(float), stream>>>(
        partial_sums, partial_sums, final_sum, blocks); // targets ignored? We'll create a separate kernel for final reduction to avoid confusion.
    // Actually, we can write a simple kernel that sums an array into one value.
    // But it's easier to call cudaMemcpy to host for partial_sums and sum on host, but that requires synchronization. Since we are returning a float, we can do cudaMemcpy and then sum host-side. This is acceptable for this use case (numel=33 million, blocks up to 1024, so copying 1024 floats to host is fine).
    float host_sum = 0.0f;
    cudaMemcpyAsync(&host_sum, final_sum, sizeof(float), cudaMemcpyDeviceToHost, stream); // Actually need to sum partial sums first.
    // Better: avoid second kernel, just copy partial_sums to host and sum on host.
    cudaFree(partial_sums);
    cudaFree(final_sum);
    // Wait, we need to synchronize to ensure copy completes before returning? The caller (binding) will synchronize if needed, but we return a float by value, so we must ensure the value is ready. We'll do a synchronous cudaMemcpy for partial_sums to a host array, then sum.
    // We'll allocate host array of blocks floats, copy, sum, free.
    // But the cudaMalloc/cudaFree inside launcher may be slow, but only one call per forward. Acceptable.
    // We'll implement a simpler approach: use thrust::reduce? Not allowed third-party, but we can use cub? No, just do our own.
    // Since numel is large, we can also do a grid-stride loop and atomicAdd to a single device variable, but atomics on global memory are slow for many blocks. With 1024 blocks, it's okay but may be contention. We'll do the two-stage reduction but implement it cleanly.
    // We'll define a helper kernel to reduce an array to a single value using one block.
    // However, to avoid extra kernel launch, we can use a cooperative groups or just use cudaMemcpy to host after the first kernel, sum on host, and return. This is simpler and correct.
    float* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    bce_loss_reduce_kernel<<<blocks, threads, threads * sizeof(float), stream>>>(logits, targets, d_partial, numel);
    float h_partial[blocks]; // cannot dynamically allocate on stack with variable size, so use std::vector? But we are in C++, we can use new/delete.
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
