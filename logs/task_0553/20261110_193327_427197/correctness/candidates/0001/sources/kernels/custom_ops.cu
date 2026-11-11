// Custom CUDA kernels for embedding and avg_pool2d
#include <cuda_runtime.h>

// Embedding kernel: indices are int64_t, weight is float, output is float
__global__ void embedding_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int num_indices,
    int embedding_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;
    int64_t token = indices[idx];
    // Note: assumes token is within [0, num_embeddings), which is guaranteed by contract
    // but we can add a safety clamp to avoid out-of-bounds if needed
    const float* src = weight + token * embedding_dim;
    float* dst = output + idx * embedding_dim;
    for (int i = 0; i < embedding_dim; ++i) {
        dst[i] = src[i];
    }
}

// AvgPool2d kernel for 4D input (N, H, W, C), pool on H,W with kernel_size, stride, padding=0
__global__ void avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w) {
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    // decompose idx into (n, c, ho, wo)
    int wo = idx % W_out;
    int tmp = idx / W_out;
    int ho = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    float sum = 0.0f;
    int count = 0;
    
    // input base pointer for (n, h, w, c)
    const float* in_base = input + ((int64_t)n * H + 0) * W * C + c;
    
    for (int kh = 0; kh < kernel_h; ++kh) {
        int h_in = ho * stride_h + kh;
        if (h_in >= H) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int w_in = wo * stride_w + kw;
            if (w_in >= W) continue;
            sum += in_base[(h_in * W + w_in) * C];
            ++count;
        }
    }
    
    // Average (count should always be kernel_h*kernel_w for valid region, but handle boundary)
    output[idx] = (count > 0) ? sum / count : 0.0f;
}

// Launcher for embedding: indices is CPU pointer, weight/output are GPU pointers
extern "C" void embedding_launcher(
    const int64_t* indices_cpu,
    const float* weight_gpu,
    float* output_gpu,
    int num_indices,
    int embedding_dim,
    cudaStream_t stream) {
    if (num_indices <= 0) return;
    
    // Allocate temporary device buffer for indices
    int64_t* indices_gpu = nullptr;
    cudaError_t err = cudaMalloc(&indices_gpu, num_indices * sizeof(int64_t));
    if (err != cudaSuccess) return; // binding should check but we cannot report; assume success
    
    // Copy CPU indices to GPU asynchronously on the provided stream
    cudaMemcpyAsync(indices_gpu, indices_cpu, num_indices * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);
    
    // Launch embedding kernel
    int threads = 256;
    int blocks = (num_indices + threads - 1) / threads;
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        indices_gpu, weight_gpu, output_gpu, num_indices, embedding_dim);
    
    // Free temporary buffer after kernel completion (cudaFree is blocking? Better to enqueue)
    // Since cudaFree synchronizes? Actually cudaFree does not synchronize by default.
    // To ensure safe free, we could use cudaFreeAsync if available, but cudaFree is acceptable
    // for simplicity. However, using cudaFree after launch without synchronization may cause issues
    // if kernel is still running. Better to use cudaFreeAsync (CUDA 11.2+) or schedule a callback.
    // We'll use cudaFreeAsync if available, else fallback to cudaFree (which may wait).
#if CUDART_VERSION >= 11020
    cudaFreeAsync(indices_gpu, stream);
#else
    cudaFree(indices_gpu); // This will synchronize on legacy CUDA, but acceptable for correctness
#endif
}

// Launcher for avg_pool2d
extern "C" void avg_pool2d_launcher(
    const float* input_gpu,
    float* output_gpu,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    cudaStream_t stream) {
    int total = N * C * H_out * W_out;
    if (total <= 0) return;
    
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input_gpu, output_gpu,
        N, C, H, W, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w);
}
