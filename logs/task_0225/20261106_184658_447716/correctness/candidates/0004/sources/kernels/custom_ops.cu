#include <cuda_runtime.h>
#include <cstdint>

// CUDA kernel to compute mean and biased variance per (N,C) over spatial dimensions H*W.
// Each block handles one (N,C) pair, using block reduction.
__global__ void instance_norm_stats_kernel(const float* __restrict__ x, float* __restrict__ mean, float* __restrict__ var, int N, int C, int HW) {
    int nc = blockIdx.x; // N*C index
    const float* x_nc = x + (size_t)nc * HW;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int i = tid; i < HW; i += stride) {
        float val = x_nc[i];
        sum += val;
        sumsq += val * val;
    }

    // Block reduction for sum and sumsq
    extern __shared__ float sdata[]; // [0..blockDim.x-1] sum, [blockDim.x..2*blockDim.x-1] sumsq
    sdata[tid] = sum;
    sdata[blockDim.x + tid] = sumsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[blockDim.x + tid] += sdata[blockDim.x + tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float total_sum = sdata[0];
        float total_sumsq = sdata[blockDim.x];
        float mean_val = total_sum / HW;
        float var_val = total_sumsq / HW - mean_val * mean_val;
        mean[nc] = mean_val;
        var[nc] = var_val;
    }
}

// CUDA kernel to normalize and apply affine.
__global__ void instance_norm_apply_kernel(const float* __restrict__ x, float* __restrict__ y, const float* __restrict__ mean, const float* __restrict__ var, const float* __restrict__ weight, const float* __restrict__ bias, float eps, int N, int C, int HW) {
    int nc = blockIdx.x; // N*C index
    const float* x_nc = x + (size_t)nc * HW;
    float* y_nc = y + (size_t)nc * HW;
    float m = mean[nc];
    float v = var[nc];
    float inv_std = rsqrtf(v + eps);
    float w = (weight != nullptr) ? weight[nc % C] : 1.0f;
    float b = (bias != nullptr) ? bias[nc % C] : 0.0f;

    int tid = threadIdx.x;
    int stride = blockDim.x;
    for (int i = tid; i < HW; i += stride) {
        float val = x_nc[i];
        float norm = (val - m) * inv_std;
        y_nc[i] = norm * w + b;
    }
}

// Launcher: computes instance norm.
// x, y: pointers to float32 tensors of shape [N,C,H,W], contiguous or with strides? We assume contiguous for simplicity.
// Actually input may have strides; in binding we call .contiguous() before passing, but here we assume contiguous and use flat indexing.
// weight, bias: pointers to C floats or nullptr.
// eps: float.
// stream: CUDA stream.
extern "C" void instance_norm_forward_launcher(const float* x, float* y, const float* weight, const float* bias, float eps, int N, int C, int HW, cudaStream_t stream) {
    int total_nc = N * C;
    int threads = 256;
    if (HW > 1024*1024) threads = 512; // tune based on HW
    int shared_bytes = 2 * threads * sizeof(float);

    // Allocate temporary storage for mean and var on device
    float* d_mean;
    float* d_var;
    cudaMalloc(&d_mean, total_nc * sizeof(float));
    cudaMalloc(&d_var, total_nc * sizeof(float));

    // Launch stats kernel
    instance_norm_stats_kernel<<<total_nc, threads, shared_bytes, stream>>>(x, d_mean, d_var, N, C, HW);

    // Launch apply kernel
    instance_norm_apply_kernel<<<total_nc, threads, 0, stream>>>(x, y, d_mean, d_var, weight, bias, eps, N, C, HW);

    // Free temporary buffers (maybe asynchronous? use cudaFree after synchronization? For simplicity, free after launch; note cudaFree may be called before kernel finishes? Usually cudaFree is safe after previous kernels complete? Better to synchronize or use stream-ordered allocation. We'll do cudaFree, but it might cause issues. Since this is launcher, we can rely on the caller to synchronize? Better: use cudaMallocAsync with stream? Simpler: allocate with cudaMalloc and free after launch, but need to ensure kernel done. We'll call cudaFree immediately after launch; however, the CUDA runtime may defer the free until stream work completes if using stream-ordered semantics with cudaFree? Actually cudaFree is synchronous and will wait for all preceding work on the device? Not guaranteed. Safer: allocate with cudaMalloc, launch kernels, then cudaStreamSynchronize(stream), then cudaFree. But that blocks. Or use cudaMallocAsync with stream, and cudaFreeAsync. Since we are not allowed to know PyTorch's allocation? We can use cudaMallocAsync from CUDA 11.2+. But to keep compatibility, we can use a simple approach: allocate temp, launch, then cudaFree after a cudaStreamSynchronize. The binding will call this launcher and then return; the synchronize will slow down but correctness first. We'll include cudaStreamSynchronize(stream) before cudaFree.

    cudaStreamSynchronize(stream);
    cudaFree(d_mean);
    cudaFree(d_var);
}
