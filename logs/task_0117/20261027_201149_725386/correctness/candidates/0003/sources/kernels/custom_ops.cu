#include <cuda_runtime.h>
#include <math.h>

// AvgPool1d forward kernel
__global__ void avgpool1d_kernel(const float* __restrict__ x,
                                 float* __restrict__ y,
                                 int batch,
                                 int channels,
                                 int length,
                                 int pooled_length,
                                 int kernel_size,
                                 int stride,
                                 int padding) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * channels * pooled_length;
    if (idx >= total) return;

    int ow = idx % pooled_length;
    int oc = (idx / pooled_length) % channels;
    int ob = idx / (pooled_length * channels);

    int start = ow * stride - padding;
    float sum = 0.0f;
    for (int k = 0; k < kernel_size; ++k) {
        int pos = start + k;
        if (pos >= 0 && pos < length) {
            sum += x[((ob * channels + oc) * length) + pos];
        }
    }
    y[idx] = sum / kernel_size;
}

// Tanh forward kernel (actually tangent)
__global__ void tanh_kernel(const float* __restrict__ x,
                            float* __restrict__ y,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = tanf(x[idx]);
    }
}

// Cross-entropy loss kernel with mean reduction
__global__ void cross_entropy_loss_kernel(const float* __restrict__ x,
                                          const long long* __restrict__ target,
                                          float* __restrict__ loss,
                                          int batch,
                                          int classes) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid >= batch) return;

    const float* sample = x + bid * classes;
    long long t = target[bid];
    
    // Find max for numerical stability
    float max_val = sample[0];
    for (int i = tid; i < classes; i += blockDim.x) {
        float v = sample[i];
        if (v > max_val) max_val = v;
    }
    // block reduction for max
    sdata[tid] = max_val;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
        }
        __syncthreads();
    }
    float block_max = sdata[0];
    __syncthreads();

    // Compute sum of exp(x_i - max)
    float sum_exp = 0.0f;
    for (int i = tid; i < classes; i += blockDim.x) {
        sum_exp += expf(sample[i] - block_max);
    }
    sdata[tid] = sum_exp;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    float block_sum_exp = sdata[0];

    // Compute loss for this sample: -x_target + log(sum_exp) + max
    float sample_loss = 0.0f;
    if (tid == 0) {
        if (t >= 0 && t < classes) {
            sample_loss = -sample[t] + logf(block_sum_exp) + block_max;
        } else {
            sample_loss = logf(block_sum_exp) + block_max;
        }
        atomicAdd(loss, sample_loss);
    }
}

// Scale kernel for loss mean reduction
__global__ void scale_kernel(float* loss, float scale) {
    *loss *= scale;
}

// Launchers
extern "C" void avgpool1d_forward_launcher(const float* x,
                                            float* y,
                                            int batch,
                                            int channels,
                                            int length,
                                            int pooled_length,
                                            int kernel_size,
                                            int stride,
                                            int padding,
                                            cudaStream_t stream) {
    int total = batch * channels * pooled_length;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    avgpool1d_kernel<<<blocks, threads, 0, stream>>>(x, y, batch, channels, length, pooled_length,
                                                    kernel_size, stride, padding);
}

extern "C" void tanh_forward_launcher(const float* x, float* y, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    tanh_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}

// Launcher for cross entropy loss (including mean scaling)
extern "C" void cross_entropy_loss_forward_launcher(const float* x,
                                                    const long long* target,
                                                    float* loss,
                                                    int batch,
                                                    int classes,
                                                    cudaStream_t stream) {
    // Initialize loss to 0 on device
    cudaMemsetAsync(loss, 0, sizeof(float), stream);

    int threads = 256;
    int blocks = batch; // one block per sample
    size_t shared_mem = threads * sizeof(float);
    cross_entropy_loss_kernel<<<blocks, threads, shared_mem, stream>>>(x, target, loss, batch, classes);

    // Scale by 1/batch for mean reduction
    float scale = 1.0f / batch;
    scale_kernel<<<1, 1, 0, stream>>>(loss, scale);
}
