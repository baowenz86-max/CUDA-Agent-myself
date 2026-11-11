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
    const float* src = weight + token * embedding_dim;
    float* dst = output + idx * embedding_dim;
    for (int i = 0; i < embedding_dim; ++i) {
        dst[i] = src[i];
    }
}

// AvgPool2d kernel: input NCHW, output NCHW
__global__ void avg_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w) {
    int total = N * C * H_out * W_out;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    
    int wo = idx % W_out;
    int tmp = idx / W_out;
    int ho = tmp % H_out;
    tmp /= H_out;
    int c = tmp % C;
    int n = tmp / C;
    
    float sum = 0.0f;
    int count = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
        int hi = ho * stride_h + kh - padding_h;
        if (hi < 0 || hi >= H) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int wi = wo * stride_w + kw - padding_w;
            if (wi < 0 || wi >= W) continue;
            sum += input[((n * C + c) * H + hi) * W + wi];
            ++count;
        }
    }
    output[idx] = (count > 0) ? sum / count : 0.0f;
}

// Launcher for embedding: accepts indices pointer that may be CPU or GPU, determined at runtime.
extern "C" void embedding_launcher(
    const int64_t* indices_ptr,
    const float* weight_gpu,
    float* output_gpu,
    int num_indices,
    int embedding_dim,
    cudaStream_t stream) {
    if (num_indices <= 0) return;
    
    int64_t* indices_gpu = nullptr;
    bool is_device_ptr = false;
    cudaPointerAttributes attrs;
    cudaError_t err = cudaPointerGetAttributes(&attrs, indices_ptr);
    if (err == cudaSuccess && attrs.type == cudaMemoryTypeDevice) {
        is_device_ptr = true;
    }
    
    if (is_device_ptr) {
        indices_gpu = const_cast<int64_t*>(indices_ptr);
    } else {
        // Assume CPU pointer
        err = cudaMalloc(&indices_gpu, num_indices * sizeof(int64_t));
        if (err != cudaSuccess) return;
        cudaMemcpyAsync(indices_gpu, indices_ptr, num_indices * sizeof(int64_t),
                        cudaMemcpyHostToDevice, stream);
    }
    
    int threads = 256;
    int blocks = (num_indices + threads - 1) / threads;
    embedding_kernel<<<blocks, threads, 0, stream>>>(
        indices_gpu, weight_gpu, output_gpu, num_indices, embedding_dim);
    
    if (!is_device_ptr) {
#if CUDART_VERSION >= 11020
        cudaFreeAsync(indices_gpu, stream);
#else
        cudaFree(indices_gpu);
#endif
    }
}

// Launcher for avg_pool2d (NCHW)
extern "C" void avg_pool2d_launcher(
    const float* input_gpu,
    float* output_gpu,
    int N, int C, int H, int W,
    int H_out, int W_out,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w,
    cudaStream_t stream) {
    int total = N * C * H_out * W_out;
    if (total <= 0) return;
    
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avg_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input_gpu, output_gpu,
        N, C, H, W, H_out, W_out,
        kernel_h, kernel_w, stride_h, stride_w,
        padding_h, padding_w);
}
