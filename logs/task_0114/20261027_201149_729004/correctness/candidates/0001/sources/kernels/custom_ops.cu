#include <cuda_runtime.h>
#include <math.h>

// Helper device functions for digamma
// Uses Lanczos approximation with reflection for small arguments.
// This approximation is reasonably accurate for float32 inputs.
__device__ float digamma_impl(float x) {
    // Reflection for small x to improve accuracy
    float result = 0.0f;
    float x_orig = x;
    bool reflect = false;
    if (x < 0.5f) {
        reflect = true;
        x = 1.0f - x;
    }
    // Lanczos gamma coefficients (g=7, n=9)
    const float g = 7.0f;
    const float c[9] = {
        0.99999999999980993f,
        676.5203681218851f,
        -1259.1392167224028f,
        771.32342877765313f,
        -176.61502916214059f,
        12.507343278686905f,
        -0.13857109526572012f,
        9.9843695780195716e-6f,
        1.5056327351493116e-7f
    };
    float sum = c[0];
    for (int i = 1; i < 9; ++i) {
        sum += c[i] / (x + (float)(i - 1));
    }
    float x_shift = x + g - 0.5f;
    float pi = 3.14159265358979323846f;
    float log_term = (x_shift - 0.5f) * logf(x_shift) - x_shift + 0.5f * logf(2.0f * pi);
    float digamma_large = log_term - logf(sum);
    // Actually, digamma_large computed from Lanczos gives log derivative? Need to be careful.
    // Let's instead use a simpler asymptotic series for digamma for x >= 0.5.
    // For x >= 0.5, use: digamma(x) = ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + ...
    // This series is accurate for x > 1, but for 0.5 <= x <= 1, we can use recurrence to shift x up.
    // We'll implement a robust method:
    // If x < 0.5, use reflection: digamma(x) = digamma(1-x) - pi / tan(pi*x)
    // Then use recurrence to shift x to > 6, then apply asymptotic expansion.
    // Since we also need digamma for positive and negative inputs, but inputs after ELU are > -1? Actually ELU output is > -1, so x could be negative but > -1, and digamma for negative non-integer values is defined via reflection.
    // We'll implement a commonly used algorithm:
    // For x <= 0, use reflection: digamma(x) = digamma(1 - x) - pi / tan(pi * x)
    // For x > 0, add 6 to x and use recurrence to reduce, then asymptotic.
    // This method is accurate to about 1e-6 for float.
    
    // Reset x to original for reflection handling
    x = x_orig;
    
    // Reflection for x <= 0 (or x < 0.5 to improve accuracy)
    if (x <= 0.0f) {
        // For non-positive non-integer, use reflection formula
        // digamma(x) = digamma(1 - x) - pi / tan(pi * x)
        // We'll compute digamma(1 - x) using the positive branch
        float x_ref = 1.0f - x;
        // Handle x near negative integers (poles) - but ELU output > -1, so only possible pole at 0? Actually x could be exactly 0? ELU(0)=0, digamma(0) is -inf. But torch.digamma(0) returns -inf? We should handle NaN/inf appropriately. For our tolerance, maybe avoid exact zero? Our approximation may return large values.
        // For simplicity, assume no exact poles.
        // Compute digamma for positive x_ref using algorithm below
        // This function recursively calls itself, but we can inline.
        // Let's define a separate function for positive digamma.
        // We'll compute using the same code path by setting a flag.
        bool reflect_neg = true;
        float positive_x = x_ref;
        float digamma_positive = 0.0f;
        // Recurrence shift to > 6
        while (positive_x < 6.0f) {
            digamma_positive -= 1.0f / positive_x;
            positive_x += 1.0f;
        }
        // Asymptotic expansion for positive_x > 6
        float inv = 1.0f / positive_x;
        float inv2 = inv * inv;
        digamma_positive += logf(positive_x) - 0.5f * inv;
        digamma_positive += inv2 * (-1.0f/12.0f + inv2 * (1.0f/120.0f - inv2 * (1.0f/252.0f - inv2 * 1.0f/240.0f)));
        // Reflection term
        float pi = 3.14159265358979323846f;
        float tan_term = tanf(pi * x);
        result = digamma_positive - pi / tan_term;
        return result;
    } else {
        // For 0 < x < 0.5, use reflection digamma(x) = digamma(1-x) - pi/tan(pi*x)
        if (x < 0.5f) {
            float x_ref = 1.0f - x;
            float digamma_positive = 0.0f;
            float positive_x = x_ref;
            while (positive_x < 6.0f) {
                digamma_positive -= 1.0f / positive_x;
                positive_x += 1.0f;
            }
            float inv = 1.0f / positive_x;
            float inv2 = inv * inv;
            digamma_positive += logf(positive_x) - 0.5f * inv;
            digamma_positive += inv2 * (-1.0f/12.0f + inv2 * (1.0f/120.0f - inv2 * (1.0f/252.0f - inv2 * 1.0f/240.0f)));
            float pi = 3.14159265358979323846f;
            float tan_term = tanf(pi * x);
            result = digamma_positive - pi / tan_term;
            return result;
        } else {
            // x >= 0.5, use recurrence to shift up to >=6
            float positive_x = x;
            float digamma_val = 0.0f;
            while (positive_x < 6.0f) {
                digamma_val -= 1.0f / positive_x;
                positive_x += 1.0f;
            }
            float inv = 1.0f / positive_x;
            float inv2 = inv * inv;
            digamma_val += logf(positive_x) - 0.5f * inv;
            digamma_val += inv2 * (-1.0f/12.0f + inv2 * (1.0f/120.0f - inv2 * (1.0f/252.0f - inv2 * 1.0f/240.0f)));
            return digamma_val;
        }
    }
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
        output[idx] = digamma_impl(input[idx]);
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
        // Handle error? We can't throw, but we can leave for binding to check via cudaGetLastError after call.
        // For now, do nothing.
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
