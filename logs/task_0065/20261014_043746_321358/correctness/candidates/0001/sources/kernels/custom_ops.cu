#include <cuda_runtime.h>

// channel_mean_forward: compute mean over dims 1,2,3 for each batch element.
// Input: x float32 shape [N,C,H,W] contiguous; Output: mean float32 shape [N,1,1,1]
__global__ void channel_mean_kernel(const float* __restrict__ x, float* __restrict__ mean, int N, int inner) {
    int n = blockIdx.x;
    float sum = 0.0f;
    for (int idx = threadIdx.x; idx < inner; idx += blockDim.x) {
        sum += x[(long long)n * inner + idx];
    }
    // block reduce
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        mean[n] = sdata[0] / (float)inner;
    }
}

extern "C" void channel_mean_launcher(const float* x, float* mean, int N, int C, int H, int W, cudaStream_t stream) {
    int inner = C * H * W;
    int threads = 256;
    dim3 grid(N);
    channel_mean_kernel<<<grid, threads, 0, stream>>>(x, mean, N, inner);
}

// subtract_mean_forward: output = x - mean - scalar
// Input: x [N,C,H,W], mean [N,1,1,1], scalar float
__global__ void subtract_mean_kernel(const float* __restrict__ x, const float* __restrict__ mean, float subtract_value, float* __restrict__ out, long long total) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int n = (int)(idx / (((long long)C) * H * W)); // wait C,H,W not captured; recalc or pass inner
        // Pass inner as parameter instead for efficiency; but kernel defined with inner maybe better.
        // Since inner = C*H*W, we can pass inner explicitly.
    }
}

// Redefine subtract kernel with inner parameter to avoid code duplication
__global__ void subtract_mean_kernel_v2(const float* __restrict__ x, const float* __restrict__ mean, float subtract_value, float* __restrict__ out, long long total, int inner) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int n = (int)(idx / inner);
        out[idx] = x[idx] - mean[n] - subtract_value;
    }
}

extern "C" void subtract_mean_launcher(const float* x, const float* mean, float subtract_value, float* out, int N, int C, int H, int W, cudaStream_t stream) {
    long long total = (long long)N * C * H * W;
    int inner = C * H * W;
    int threads = 256;
    dim3 grid((total + threads - 1) / threads);
    subtract_mean_kernel_v2<<<grid, threads, 0, stream>>>(x, mean, subtract_value, out, total, inner);
}
