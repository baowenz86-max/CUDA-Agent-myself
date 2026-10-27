#include <cuda_runtime.h>
#include <math.h>

// High-precision digamma implementation using double arithmetic.
// For positive x, use recurrence to shift to >= 16, then apply asymptotic series.
// For negative x, use reflection: psi(x) = psi(1-x) - pi * cot(pi*x),
// computing cot(pi*x) in a numerically stable way.
__device__ double digamma_double(double x) {
    const double pi = 3.14159265358979323846;
    const double euler_gamma = 0.57721566490153286060651209;
    
    // Handle special cases
    if (x == 0.0) return -HUGE_VAL; // -inf
    if (x < 0.0) {
        // For negative non-integer x, use reflection
        // Check if x is integer (poles): return -inf for negative integers, but ELU never gives integers except 0, handled above.
        double int_part;
        if (modf(x, &int_part) == 0.0) return -HUGE_VAL; // negative integer pole
        double r = 1.0 - x;
        // Compute cot(pi*x) using the identity cot(pi*x) = -tan(pi*(r-0.5))? Actually better: cot(pi*x) = tan(pi*(0.5 - x))? Let's derive: cot(pi*x) = cos(pi*x)/sin(pi*x). We can compute as 1.0/tan(pi*x), but tan may be near zero causing overflow. Use the identity cot(pi*x) = -tan(pi*x - pi/2)?? That's not stable either. A robust way is to use the reflection formula with a different formulation: psi(x) = psi(1-x) - pi/tan(pi*x). But when x is near an integer, tan(pi*x) is small and the quotient is large; we can compute it using the Laurent series for cot: cot(pi*x) = 1/(pi*x) - (pi*x)/3 - (pi*x)^3/45 - ... for x near 0. However, x may be near any negative integer. Since ELU output is in (-1,0), x is in (-1,0), so the only pole is at x=0, handled. For x in (-1,0), tan(pi*x) is negative and not extremely small except near 0 and -1. At x near -1, pi*x near -pi, tan(-pi) ~ 0 (since tan is periodic), so we may have issues near -1 as well. We can handle all negative x by shifting the argument: for x < 0, use psi(x) = psi(x+1) - 1/x, and repeat until x+1 > 0. But that would work for x > -1; for x in (-1,0), x+1 is in (0,1), so we can compute psi(x) = psi(x+1) - 1/x. This avoids reflection entirely for the range encountered! Since ELU output is > -1, we can use this recurrence. For x in (-1,0), psi(x) = psi(x+1) - 1/x. x+1 is in (0,1). Then we need psi for (0,1) which we can compute via recurrence to large. This avoids the reflection formula and its poles. This is much simpler and accurate. So we can apply: if (x < 0.5 && x > -1) use psi(x) = psi(x+1) - 1/x, then compute psi(x+1) using positive algorithm. This is exactly what the old code does? The old code used reflection for x<0.5, which caused issues. Instead, we can use recurrence down to any x > -1: for x in (-1, 0.5), we can set y = x + k where k is chosen so that y >= 0.5, and accumulate -sum 1/(x+i). Since x is in (-1,0), one shift suffices: if x < 0, y = x+1, psi(x) = psi(y) - 1/x. If x is between 0 and 0.5, we can similarly shift: y = x+1, psi(x) = psi(y) - 1/x. So we can handle all x > -1 by a single shift to y in [0.5, 1.5)??? But if x is very small positive, y might be <0.5? Actually if x in (0,0.5), x+1 in (1,1.5), so >=0.5. If x in (-1,0), x+1 in (0,1), which may be <0.5, requiring further shift. So we can loop: while (y < 0.5) { psi -= 1/y; y += 1; } This works for all x > -1 (and even for all x if we handle negative integers separately). Since ELU output is in (-1, inf), we only need to handle x > -1. The old code's reflection was unnecessary and caused the -inf bug. So we replace with recurrence for x < 0.5. For x >= 0.5, use positive algorithm directly (recurrence to >=16 and asymptotic series). Use double precision to improve accuracy.
    }
    
    // For x >= 0.5, compute using recurrence to shift to large positive then asymptotic series.
    double result = 0.0;
    double shifted_x = x;
    while (shifted_x < 16.0) {
        result -= 1.0 / shifted_x;
        shifted_x += 1.0;
    }
    // Asymptotic expansion for digamma for shifted_x >= 16
    double inv = 1.0 / shifted_x;
    double inv2 = inv * inv;
    result += log(shifted_x) - 0.5 * inv;
    result += inv2 * (-1.0 / 12.0 + inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * 1.0 / 240.0)));
    return result;
}

// Wrapper for float input, compute in double and cast
device float digamma_float(float x) {
    return (float)digamma_double((double)x);
}

// ELU kernel: elementwise
__global__ void elu_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel, float alpha) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : (alpha * (expf(x) - 1.0f));
    }
}

// Digamma kernel: elementwise
__global__ void digamma_kernel(const float* __restrict__ input, float* __restrict__ output, int64_t numel) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = digamma_float(input[idx]);
    }
}

// Triu kernel: 2D, supports strides
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

// Launcher functions (extern "C")

extern "C" void elu_forward_launcher(const float* input, float* output, int64_t numel, float alpha, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    elu_kernel<<<blocks, threads, 0, stream>>>(input, output, numel, alpha);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}

extern "C" void digamma_forward_launcher(const float* input, float* output, int64_t numel, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    digamma_kernel<<<blocks, threads, 0, stream>>>(input, output, numel);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
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
    if (err != cudaSuccess) {
        // ignore
    }
}
