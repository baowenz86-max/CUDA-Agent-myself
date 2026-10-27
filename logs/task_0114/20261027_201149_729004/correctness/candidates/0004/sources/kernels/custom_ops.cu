#include <cuda_runtime.h>
#include <math.h>

__device__ double digamma_double(double x) {
    const double PI = 3.14159265358979323846;
    // Reflection for x <= 0.5
    if (x <= 0.5) {
        // Use reflection: psi(x) = psi(1-x) - pi / tan(pi*x)
        // For x close to 0 or negative integers, this may produce inf/nan, but those cases are rare.
        double x_ref = 1.0 - x;
        double tan_term = tan(PI * x);
        double psi_ref = digamma_double(x_ref);
        return psi_ref - PI / tan_term;
    }
    // For x > 0.5, use recurrence to shift to x >= 16
    double result = 0.0;
    double cur_x = x;
    while (cur_x < 16.0) {
        result -= 1.0 / cur_x;
        cur_x += 1.0;
    }
    // Asymptotic expansion for cur_x >= 16
    double inv = 1.0 / cur_x;
    double inv2 = inv * inv;
    result += log(cur_x) - 0.5 * inv;
    result += inv2 * (-1.0/12.0 + inv2 * (1.0/120.0 - inv2 * (1.0/252.0 - inv2 * (1.0/240.0 - inv2 * 5.0/660.0))));
    return result;
}

__global__ void digamma_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        double x = (double)input[idx];
        output[idx] = (float)digamma_double(x);
    }
}

// ELU kernel (unchanged, but included for completeness)
__global__ void elu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel, float alpha) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : (alpha * (expf(x) - 1.0f));
    }
}

// Triu kernel (unchanged)
__global__ void triu_kernel(const float* __restrict__ input, float* __restrict__ output,
                            int64_t rows, int64_t cols,
                            int64_t in_stride_row, int64_t in_stride_col,
                            int64_t out_stride_row, int64_t out_stride_col) {
    int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        int64_t in_idx = row * in_stride_row + col * in_stride_col;
        int64_t out_idx = row * out_stride_row + col * out_stride_col;
        if (col >= row) {
            output[out_idx] = input[in_idx];
        } else {
            output[out_idx] = 0.0f;
        }
    }
}

// Launchers
extern "C" void elu_forward_launcher(const float* input, float* output, int64_t numel, float alpha, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    elu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, alpha);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* ignore */ }
}

extern "C" void digamma_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    digamma_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* ignore */ }
}

extern "C" void triu_forward_launcher(const float* input, float* output,
                                    int64_t rows, int64_t cols,
                                    int64_t in_stride_row, int64_t in_stride_col,
                                    int64_t out_stride_row, int64_t out_stride_col,
                                    cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks((cols + 15) / 16, (rows + 15) / 16);
    triu_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols,
                                                in_stride_row, in_stride_col,
                                                out_stride_row, out_stride_col);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { /* ignore */ }
}
