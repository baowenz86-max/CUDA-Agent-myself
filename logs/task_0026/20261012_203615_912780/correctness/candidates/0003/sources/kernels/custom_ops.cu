// CUDA kernels for one_hot, maximum_scalar, bucketize
#include <cuda_runtime.h>

// Kernel: one_hot_forward
// Input: x (int64) shape [B, L]
// Output: out (float) shape [B, L, C]
__global__ void one_hot_kernel(const long long* __restrict__ x,
                               float* __restrict__ out,
                               long long num_elements,
                               int num_classes,
                               int L,
                               int C) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (; idx < num_elements; idx += stride) {
        long long b = idx / L;
        long long l = idx % L;
        long long cls = x[idx];
        long long out_idx = (b * L + l) * C + cls;
        out[out_idx] = 1.0f;
    }
}

// Kernel: maximum_scalar_forward
// Input: x (float) shape [B, L, C]
// Output: out (float) shape [B, L, C]
// scalar: 0.0
__global__ void maximum_scalar_kernel(const float* __restrict__ x,
                                      float* __restrict__ out,
                                      long long num_elements) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (; idx < num_elements; idx += stride) {
        float v = x[idx];
        out[idx] = (v > 0.0f) ? v : 0.0f;
    }
}

// Kernel: bucketize_forward
// Input: values (float) shape [B, L]
// boundaries: float 1D ascending, length nb
// Output: out (int64) shape [B, L], right=False
__global__ void bucketize_kernel(const float* __restrict__ values,
                                 const float* __restrict__ boundaries,
                                 long long* __restrict__ out,
                                 long long num_elements,
                                 int nb) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = blockDim.x * gridDim.x;
    for (; idx < num_elements; idx += stride) {
        float val = values[idx];
        int lo = 0, hi = nb;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (boundaries[mid] < val) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[idx] = (long long)lo;
    }
}

// Launcher: one_hot_forward
// x: int64 pointer, shape [B, L]
// num_elements: B*L
// num_classes: C
// L: sequence length
// stream: CUDA stream
extern "C" void one_hot_forward_launcher(const long long* x,
                                          float* out,
                                          long long num_elements,
                                          int num_classes,
                                          int L,
                                          cudaStream_t stream) {
    const int threads = 256;
    const long long blocks_ll = (num_elements + threads - 1) / threads;
    const int max_grid = 65535;
    int blocks = (int)min((long long)max_grid, blocks_ll);
    one_hot_kernel<<<blocks, threads, 0, stream>>>(x, out, num_elements, num_classes, L, num_classes);
}

// Launcher: maximum_scalar_forward
// x: float pointer, shape [B, L, C]
// num_elements: B*L*C
extern "C" void maximum_scalar_forward_launcher(const float* x,
                                                  float* out,
                                                  long long num_elements,
                                                  cudaStream_t stream) {
    const int threads = 256;
    const long long blocks_ll = (num_elements + threads - 1) / threads;
    const int max_grid = 65535;
    int blocks = (int)min((long long)max_grid, blocks_ll);
    maximum_scalar_kernel<<<blocks, threads, 0, stream>>>(x, out, num_elements);
}

// Launcher: bucketize_forward
// values: float pointer, shape [B, L]
// boundaries: float pointer, length nb
// out: int64 pointer, shape [B, L]
// num_elements: B*L
// nb: number of boundaries
extern "C" void bucketize_forward_launcher(const float* values,
                                             const float* boundaries,
                                             long long* out,
                                             long long num_elements,
                                             int nb,
                                             cudaStream_t stream) {
    const int threads = 256;
    const long long blocks_ll = (num_elements + threads - 1) / threads;
    const int max_grid = 65535;
    int blocks = (int)min((long long)max_grid, blocks_ll);
    bucketize_kernel<<<blocks, threads, 0, stream>>>(values, boundaries, out, num_elements, nb);
}