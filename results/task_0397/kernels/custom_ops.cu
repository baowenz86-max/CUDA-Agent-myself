// Softshrink CUDA kernel
#include <cuda_runtime.h>

// Vectorized kernel using float4 for improved memory bandwidth
__global__ void softshrink_kernel(float* __restrict__ output, const float* __restrict__ input, long long n, float lambd) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    // Grid-stride loop over vector elements (4 floats each)
    long long n_vec = n / 4;
    for (long long i = idx; i < n_vec; i += stride) {
        float4 in = reinterpret_cast<const float4*>(input)[i];
        float4 out;
        out.x = (in.x > lambd) ? (in.x - lambd) : ((in.x < -lambd) ? (in.x + lambd) : 0.0f);
        out.y = (in.y > lambd) ? (in.y - lambd) : ((in.y < -lambd) ? (in.y + lambd) : 0.0f);
        out.z = (in.z > lambd) ? (in.z - lambd) : ((in.z < -lambd) ? (in.z + lambd) : 0.0f);
        out.w = (in.w > lambd) ? (in.w - lambd) : ((in.w < -lambd) ? (in.w + lambd) : 0.0f);
        reinterpret_cast<float4*>(output)[i] = out;
    }
    // Handle remainder elements
    long long rem_start = n_vec * 4;
    for (long long i = rem_start + idx; i < n; i += stride) {
        float val = input[i];
        output[i] = (val > lambd) ? (val - lambd) : ((val < -lambd) ? (val + lambd) : 0.0f);
    }
}

// Launcher: raw CUDA pointers, length, lambd, stream
extern "C" void softshrink_launcher(float* output, const float* input, long long n, float lambd, cudaStream_t stream) {
    if (n <= 0) return;
    // Choose block size 256, grid based on number of vectors
    int threads = 256;
    long long n_vec = n / 4;
    long long blocks_ll = (n_vec + threads - 1) / threads;
    if (blocks_ll == 0) blocks_ll = 1;
    int blocks = (int)blocks_ll;
    softshrink_kernel<<<blocks, threads, 0, stream>>>(output, input, n, lambd);
}
