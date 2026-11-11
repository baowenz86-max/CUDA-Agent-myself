#include <cuda_runtime.h>

// Custom clamp kernel: clamps each element to [0,1]
__global__ void clamp_min_max_kernel(float* out, const float* in, long long n) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    for (long long i = idx; i < n; i += stride) {
        float v = in[i];
        out[i] = fminf(fmaxf(v, 0.0f), 1.0f);
    }
}

// Custom triu kernel: for each batch, zero out elements below main diagonal
__global__ void triu_kernel(float* out, const float* in,
                            int batch, int rows, int cols) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    long long total = (long long)batch * rows * cols;
    for (long long i = idx; i < total; i += stride) {
        int col = i % cols;
        long long tmp = i / cols;
        int row = tmp % rows;
        // batch index not needed for mask, only row and col matter
        if (row <= col) {
            out[i] = in[i];
        } else {
            out[i] = 0.0f;
        }
    }
}

// Custom add bias kernel: broadcast bias over batch dimension
__global__ void add_bias_kernel(float* out, const float* in, const float* bias,
                                int batch, int rows, int cols) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = gridDim.x * blockDim.x;
    long long total = (long long)batch * rows * cols;
    for (long long i = idx; i < total; i += stride) {
        int col = i % cols;
        long long tmp = i / cols;
        int row = tmp % rows;
        out[i] = in[i] + bias[(long long)row * cols + col];
    }
}

// Launcher for custom_clamp_min_max
extern "C" void custom_clamp_min_max_launcher(float* out, const float* in,
                                              long long n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // grid cap, grid-stride loop handles rest
    clamp_min_max_kernel<<<blocks, threads, 0, stream>>>(out, in, n);
}

// Launcher for custom_triu
extern "C" void custom_triu_launcher(float* out, const float* in,
                                     int batch, int rows, int cols,
                                     cudaStream_t stream) {
    int threads = 256;
    long long total = (long long)batch * rows * cols;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    triu_kernel<<<blocks, threads, 0, stream>>>(out, in, batch, rows, cols);
}

// Launcher for custom_add_bias
extern "C" void custom_add_bias_launcher(float* out, const float* in, const float* bias,
                                         int batch, int rows, int cols,
                                         cudaStream_t stream) {
    int threads = 256;
    long long total = (long long)batch * rows * cols;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    add_bias_kernel<<<blocks, threads, 0, stream>>>(out, in, bias, batch, rows, cols);
}
