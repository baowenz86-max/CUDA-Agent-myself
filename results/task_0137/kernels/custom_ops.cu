#include <cuda_runtime.h>
#include <cstdint>

// Forward declaration for divide_by_n_kernel
__global__ void divide_by_n_kernel(const float* sum, float* out, long long N);

// Kernel: compute mean along dimension 1 of a 4D tensor [B, C, H, W]
// Input: x [B, C, H, W] with strides sx_b, sx_c, sx_h, sx_w
// Output: out [B, H, W] contiguous (stride assumed as H*W, W, 1)
// Each thread computes one output element (b, h, w) by summing over C channels.
__global__ void mean_dim1_kernel(
    const float* __restrict__ x,
    float* __restrict__ out,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w
) {
    long long total = (long long)B * H * W;
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (; idx < total; idx += stride) {
        int w = idx % W;
        int tmp = idx / W;
        int h = tmp % H;
        int b = tmp / H;

        // Compute input base offset: b*sx_b + h*sx_h + w*sx_w
        long long base = (long long)b * sx_b + (long long)h * sx_h + (long long)w * sx_w;
        float sum = 0.0f;
        for (int c = 0; c < C; ++c) {
            long long offset = base + (long long)c * sx_c;
            sum += x[offset];
        }
        out[idx] = sum / (float)C;
    }
}

// Kernel: compute smooth L1 loss mean between two tensors of same shape [N]
// Each thread processes multiple elements (grid-stride), computes local sum of per-element loss,
// then does block reduction to get block sum, and atomicAdd to global accumulator.
__global__ void smooth_l1_loss_kernel(
    const float* __restrict__ input,
    const float* __restrict__ target,
    float* __restrict__ global_sum,
    long long N
) {
    extern __shared__ float sdata[];
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (; idx < N; idx += stride) {
        float diff = input[idx] - target[idx];
        float abs_diff = fabsf(diff);
        if (abs_diff < 1.0f) {
            local_sum += 0.5f * diff * diff;
        } else {
            local_sum += abs_diff - 0.5f;
        }
    }

    // Block reduction
    int tid = threadIdx.x;
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

// Launcher for mean_dim1: computes mean over dimension 1.
// x: input pointer, out: output pointer, B,C,H,W: dims, strides as long long,
// stream: CUDA stream to launch kernel.
extern "C" void mean_dim1_launcher(
    const float* x,
    float* out,
    int B, int C, int H, int W,
    long long sx_b, long long sx_c, long long sx_h, long long sx_w,
    cudaStream_t stream
) {
    long long total = (long long)B * H * W;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535; // grid-stride loop handles larger sizes
    if (blocks == 0) blocks = 1;

    mean_dim1_kernel<<<blocks, threads, 0, stream>>>(
        x, out, B, C, H, W, sx_b, sx_c, sx_h, sx_w
    );
}

// Launcher for smooth_l1_loss: input, target same shape N, output scalar tensor (size 1).
// Allocates temporary device memory for global sum internally, initializes to zero,
// launches kernel with atomicAdd, then divides by N on device using a small kernel.
extern "C" void smooth_l1_loss_launcher(
    const float* input,
    const float* target,
    float* out,
    long long N,
    cudaStream_t stream
) {
    // Allocate device memory for sum accumulator
    float* d_sum = nullptr;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemsetAsync(d_sum, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    if (blocks > 65535) blocks = 65535;
    if (blocks == 0) blocks = 1;

    size_t shared_mem = threads * sizeof(float);
    smooth_l1_loss_kernel<<<blocks, threads, shared_mem, stream>>>(
        input, target, d_sum, N
    );

    // Divide by N to get mean: launch a single thread kernel
    // If N == 0, resulting loss is NaN, but contract doesn't expect zero-sized tensors.
    // For safety, set to 0 if N==0.
    if (N > 0) {
        divide_by_n_kernel<<<1, 1, 0, stream>>>(d_sum, out, N);
    } else {
        // For safety, set output to 0? Actually empty tensor, but no such case. write 0.
        float zero = 0.0f;
        cudaMemcpyAsync(out, &zero, sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    // Don't free d_sum immediately as kernel may still be using it.
    // We need to synchronize or use callback. Simpler: don't free? We can use cudaFree after ensuring kernel done.
    // Since launcher returns without sync, we cannot free immediately. But we can use cudaStreamAddCallback or just allocate small and leak? Better: use cudaMallocAsync? No.
    // We'll use a static approach: allocate d_sum in global memory once, but not good for multi-stream.
    // Alternative: use cudaFree after cudaStreamSynchronize? That blocks. But acceptable for correctness? The binding may not synchronize.
    // To avoid memory leaks, we can use cudaMalloc and cudaFree with stream ordering: cudaFree is synchronous and will wait all previous operations? Actually cudaFree blocks until all preceding work on the device completes, so it's safe but may cause synchronization.
    // Given performance, we can use a small pool or just accept sync. But to keep launcher non-blocking, we could allocate d_sum using cudaMallocAsync if supported, but that requires pool. Simpler: use a __device__ variable? Not possible across kernels easily.
    // We'll use cudaMalloc and cudaFree after launching all kernels; cudaFree will implicitly synchronize? According to CUDA docs, cudaFree synchronizes with respect to the device, but may be deferred until all preceding work completes, so it acts as a synchronization point. However, it's acceptable for correctness, though may reduce performance. If we want to avoid sync, we could use cudaMemPool or persistent buffer, but too complex for this simple case.
    // Let's just use cudaMalloc and cudaFree, accepting potential implicit sync. This is okay for functionality, but may impact performance if called frequently. However, smooth_l1_loss likely not called many times per forward, so acceptable.
    // Actually we can also use a single uint32_t atomic and have the kernel compute mean directly, but need to avoid division in every block: we could have each block atomicAdd its local sum * (1/N), but then division must be done per block, and accumulated error. Better as done.
    // For simplicity, we'll free after launch with cudaFree; it will wait for all launched work.
    cudaFree(d_sum);
}

// Helper kernel to divide sum by N and write to output scalar.
__global__ void divide_by_n_kernel(const float* sum, float* out, long long N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = sum[0] / (float)N;
    }
}
