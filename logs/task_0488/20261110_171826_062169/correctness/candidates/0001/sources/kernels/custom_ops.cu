#include <cuda_runtime.h>
#include <math.h>

// Kernel for softmax along dim=1 of a 3D tensor [N, H, W]
// Each block handles one softmax vector (or multiple via grid-stride within H dimension)
// We assign one thread per H element, and each block processes one (n, w) column of H elements.
// To maximize parallelism, we can use multiple blocks per vector if H is large, but H=2048 fits in one block with 1024 threads and looping.
// Here we design a kernel where each block processes a chunk of softmax vectors, using shared memory for reductions.
// Simpler approach: each block processes one (n,w) pair and loops over H twice (max, sum).
__global__ void softmax_dim1_kernel(const float* __restrict__ input, float* __restrict__ output,
                                     int N, int H, int W) {
    int nw = blockIdx.x; // combined n and w index
    int n = nw / W;
    int w = nw % W;
    if (n >= N) return;
    
    const float* vec = input + n * H * W + w; // stride in H dimension is W, so element at h has offset h*W + w
    float* out_vec = output + n * H * W + w;
    
    // Find max
    float max_val = -INFINITY;
    for (int h = 0; h < H; ++h) {
        float val = vec[h * W];
        if (val > max_val) max_val = val;
    }
    
    // Compute sum of exp(x - max)
    float sum = 0.0f;
    for (int h = 0; h < H; ++h) {
        float val = __expf(vec[h * W] - max_val);
        sum += val;
        out_vec[h * W] = val; // store exp temporarily, will divide later
    }
    
    // Divide by sum
    float inv_sum = 1.0f / sum;
    for (int h = 0; h < H; ++h) {
        out_vec[h * W] *= inv_sum;
    }
}

// Launcher for softmax dim=1
// input: [N, H, W], output same shape, dim must be 1 (checked in binding)
extern "C" void custom_softmax_dim1_launcher(const float* input, float* output,
                                              int N, int H, int W, cudaStream_t stream) {
    int total_vectors = N * W;
    int threads = 256;
    int blocks = (total_vectors + threads - 1) / threads;
    softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, N, H, W);
}

// Kernel for KL divergence batchmean: sum(target*(log(target)-input)) / N
// Here input and target are the same tensor (softmax output).
// We can compute sum over all elements using a reduction.
__global__ void kl_div_batchmean_kernel(const float* __restrict__ x, float* __restrict__ partial_sums,
                                         int num_elements, int N) {
    // Each thread computes its element contribution
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        float val = x[idx];
        float contrib = val * (__logf(val) - val);
        // Atomic add to a single global accumulator? Better use reduction per block.
        // For simplicity, we use one block and shared memory reduction, but num_elements may be huge (32M).
        // We can use a grid-stride loop with atomicAdd to a single float, but atomicAdd on float is supported.
        // However, for performance, we'll do a two-stage reduction: each block computes partial sum via shared memory, then atomicAdd to global.
    }
}

// Better reduction implementation:
__global__ void kl_div_partial_sum_kernel(const float* __restrict__ x, float* __restrict__ partial_sums,
                                           int num_elements) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    // Grid-stride loop
    for (int idx = i; idx < num_elements; idx += blockDim.x * gridDim.x) {
        float val = x[idx];
        sum += val * (__logf(val) - val);
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
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

// Kernel to combine partial sums and divide by N
__global__ void kl_div_final_reduce_kernel(const float* __restrict__ partial_sums, float* __restrict__ output,
                                            int num_blocks, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial_sums[i];
    }
    sdata[tid] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        output[0] = sdata[0] / (float)N;
    }
}

extern "C" void custom_kl_div_batchmean_launcher(const float* x, float* output,
                                                  int num_elements, int N, cudaStream_t stream) {
    // Determine block and grid sizes for partial sum kernel
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    // Limit blocks to avoid launching too many (e.g., max 1024 blocks)
    int max_blocks = 1024;
    if (blocks > max_blocks) blocks = max_blocks;
    
    // Allocate temporary storage for partial sums (num blocks)
    // We'll need to allocate device memory. The launcher should receive a workspace pointer from binding.
    // For simplicity, we can use a single atomicAdd to a global memory accumulator, but that is slower.
    // Better: allocate workspace in binding and pass pointer. The binding will allocate via cudaMalloc.
    // We'll assume the binding passes a preallocated workspace pointer `partial_sums` of size blocks.
    // In this launcher signature, we include partial_sums pointer.
}

// We need to adjust launcher signatures: the binding should allocate workspace and pass it.
// Therefore, we define the launcher to accept a workspace pointer.
// But to keep the .cu self-contained, we can allocate using cudaMalloc inside launcher, then free after kernel.
// However, that would be slow if called often. Since this is a one-time computation, it's acceptable.
// We'll implement a launcher that allocates workspace internally and frees it.

extern "C" void custom_kl_div_batchmean_launcher(const float* x, float* output,
                                                  int num_elements, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    int max_blocks = 1024;
    if (blocks > max_blocks) blocks = max_blocks;
    
    float* partial_sums = nullptr;
    cudaError_t err = cudaMalloc(&partial_sums, blocks * sizeof(float));
    if (err != cudaSuccess) {
        // handle error: maybe set output to NaN? But binding should check error after launcher.
        // We'll just return; binding will check cudaGetLastError after launch.
        return;
    }
    
    int shared_mem = threads * sizeof(float);
    kl_div_partial_sum_kernel<<<blocks, threads, shared_mem, stream>>>(x, partial_sums, num_elements);
    
    // Final reduce: use one block
    int final_threads = 256;
    int final_shared_mem = final_threads * sizeof(float);
    kl_div_final_reduce_kernel<<<1, final_threads, final_shared_mem, stream>>>(partial_sums, output, blocks, N);
    
    cudaFree(partial_sums);
}

// For softmax, the binding can call custom_softmax_dim1_launcher directly.

// Since dim is always 1 for this task, we only implement that case.
// The binding will check dim == 1 before calling.

// Ensure no undefined symbols: we use __expf and __logf which are available in CUDA.
