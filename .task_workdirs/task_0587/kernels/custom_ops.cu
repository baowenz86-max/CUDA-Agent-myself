// kernels/custom_ops.cu
#include <cuda_runtime.h>
#include <cmath>

// Kernel to replace inf with NaN
template <typename T>
__global__ void replace_inf_with_nan_kernel(T* __restrict__ out, const T* __restrict__ in, int64_t N) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        T val = in[idx];
        if (isinf(val)) {
            out[idx] = static_cast<T>(NAN);
        } else {
            out[idx] = val;
        }
    }
}

// Kernel for group normalization where C == num_groups, so each channel is independent.
// Each block handles one (sample, group) pair, which corresponds to one channel.
// Grid: (N * C) blocks, block size = 256 threads. Each thread processes HW/256 elements.
// Shared memory stores partial sums for mean and var.
template <typename T>
__global__ void group_norm_forward_kernel(
    T* __restrict__ out,
    const T* __restrict__ x,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    int64_t N,
    int64_t C,
    int64_t HW,
    T eps) {
    
    extern __shared__ T sdata[];  // two arrays: sum and sqsum
    T* s_sum = sdata;
    T* s_sqsum = sdata + blockDim.x;
    
    int64_t sample = blockIdx.x / C;
    int64_t channel = blockIdx.x % C;
    
    // Base pointer for this sample's channel
    const T* x_ptr = x + sample * C * HW + channel * HW;
    T* out_ptr = out + sample * C * HW + channel * HW;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    T local_sum = 0;
    T local_sqsum = 0;
    
    // Grid-stride loop over HW elements
    for (int64_t i = tid; i < HW; i += stride) {
        T val = x_ptr[i];
        local_sum += val;
        local_sqsum += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sqsum[tid] = local_sqsum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_sum[tid] += s_sum[tid + offset];
            s_sqsum[tid] += s_sqsum[tid + offset];
        }
        __syncthreads();
    }
    
    T total_sum = s_sum[0];
    T total_sqsum = s_sqsum[0];
    
    T mean = total_sum / HW;
    T var = total_sqsum / HW - mean * mean;
    T inv_std = rsqrt(var + eps);
    
    T w = weight[channel];
    T b = bias[channel];
    
    // Apply normalization and affine
    for (int64_t i = tid; i < HW; i += stride) {
        T val = x_ptr[i];
        T norm = (val - mean) * inv_std;
        out_ptr[i] = w * norm + b;
    }
}

// Launcher for replace_inf_with_nan
extern "C" void replace_inf_with_nan_launcher(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream) {
    
    if (numel <= 0) return;
    
    const int threads = 256;
    int64_t blocks = (numel + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    
    replace_inf_with_nan_kernel<float><<<blocks, threads, 0, stream>>>(
        output, input, numel);
}

// Launcher for group_norm_forward
// Assumes x is contiguous float32 with shape [N, C, HW], C == num_groups
extern "C" void group_norm_forward_launcher(
    float* output,
    const float* x,
    const float* weight,
    const float* bias,
    int64_t N,
    int64_t C,
    int64_t HW,
    float eps,
    cudaStream_t stream) {
    
    if (N <= 0 || C <= 0 || HW <= 0) return;
    
    const int threads = 256;
    int blocks = N * C;
    if (blocks <= 0) return;
    
    size_t shared_mem_size = 2 * threads * sizeof(float);
    
    group_norm_forward_kernel<float><<<blocks, threads, shared_mem_size, stream>>>(
        output, x, weight, bias, N, C, HW, eps);
}
