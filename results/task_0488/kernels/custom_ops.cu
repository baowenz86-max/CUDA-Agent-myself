#include <cuda_runtime.h>
#include <math.h>

// Kernel for softmax along dim=1 of a 3D tensor [N, H, W]
// Each block handles one softmax vector (or multiple via grid-stride within H dimension)
// We assign one thread per H element, and each block processes one (n, w) column of H elements.
__global__ void softmax_dim1_kernel(const float* __restrict__ input, float* __restrict__ output,
                                     int N, int H, int W) {
    int nw = blockIdx.x; // combined n and w index
    int n = nw / W;
    int w = nw % W;
    if (n >= N) return;
    
    const float* vec = input + n * H * W + w;
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
        out_vec[h * W] = val;
    }
    
    // Divide by sum
    float inv_sum = 1.0f / sum;
    for (int h = 0; h < H; ++h) {
        out_vec[h * W] *= inv_sum;
    }
}

// Launcher for softmax dim=1
extern "C" void custom_softmax_dim1_launcher(const float* input, float* output,
                                              int N, int H, int W, cudaStream_t stream) {
    int total_vectors = N * W;
    int threads = 256;
    int blocks = (total_vectors + threads - 1) / threads;
    softmax_dim1_kernel<<<blocks, threads, 0, stream>>>(input, output, N, H, W);
}

// Kernel: partial sum per block with grid-stride loop
__global__ void kl_div_partial_sum_kernel(const float* __restrict__ x, float* __restrict__ partial_sums,
                                           int num_elements) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;
    for (int idx = i; idx < num_elements; idx += blockDim.x * gridDim.x) {
        float val = x[idx];
        sum += val * (__logf(val) - val);
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

// Single launcher for KL divergence batchmean (duplicate stub removed)
extern "C" void custom_kl_div_batchmean_launcher(const float* x, float* output,
                                                  int num_elements, int N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;
    int max_blocks = 1024;
    if (blocks > max_blocks) blocks = max_blocks;
    
    float* partial_sums = nullptr;
    cudaError_t err = cudaMalloc(&partial_sums, blocks * sizeof(float));
    if (err != cudaSuccess) {
        return;
    }
    
    int shared_mem = threads * sizeof(float);
    kl_div_partial_sum_kernel<<<blocks, threads, shared_mem, stream>>>(x, partial_sums, num_elements);
    
    int final_threads = 256;
    int final_shared_mem = final_threads * sizeof(float);
    kl_div_final_reduce_kernel<<<1, final_threads, final_shared_mem, stream>>>(partial_sums, output, blocks, N);
    
    cudaFree(partial_sums);
}
