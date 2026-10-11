#include <cuda_runtime.h>
#include <stdint.h>
#include <cuda_fp16.h>

// Kernel to zero out elements below the diagonal in each 2D slice of the input tensor.
// Input tensor shape: (N, C, H, W), treated as N*C matrices of size H x W (row-major).
// For each matrix, elements with col index < row index + diagonal are set to zero.
template<typename T>
__global__ void triu_kernel(T* output, const T* input, int num_elements, int rows, int cols, int diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // Map linear index to (batch, row, col)
    int matrix_stride = rows * cols;
    int in_matrix = idx % matrix_stride;
    int row = in_matrix / cols;
    int col = in_matrix % cols;

    // Apply upper triangular mask: keep if col >= row + diagonal
    if (col >= (row + diagonal)) {
        output[idx] = input[idx];
    } else {
        output[idx] = T(0);
    }
}

// Specialization for __half to handle zero conversion correctly
inline __device__ __half half_zero() { return __float2half(0.0f); }

template<>
__global__ void triu_kernel<__half>(__half* output, const __half* input, int num_elements, int rows, int cols, int diagonal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    int matrix_stride = rows * cols;
    int in_matrix = idx % matrix_stride;
    int row = in_matrix / cols;
    int col = in_matrix % cols;

    if (col >= (row + diagonal)) {
        output[idx] = input[idx];
    } else {
        output[idx] = half_zero();
    }
}

// Launcher function exported for binding
extern "C" void triu_forward_launcher(
    void* output,
    const void* input,
    int64_t num_elements,
    int64_t rows,
    int64_t cols,
    int64_t diagonal,
    int64_t dtype,
    cudaStream_t stream) {

    int threads = 256;
    int blocks = (int)((num_elements + threads - 1) / threads);

    // dtype: 0=float32, 1=float64, 2=half, 3=uint8, 4=int32, 5=int64, 6=bool
    switch(dtype) {
        case 0: triu_kernel<float><<<blocks, threads, 0, stream>>>(
                    (float*)output, (const float*)input, (int)num_elements, (int)rows, (int)cols, (int)diagonal); break;
        case 1: triu_kernel<double><<<blocks, threads, 0, stream>>>(
                    (double*)output, (const double*)input, (int)num_elements, (int)rows, (int)cols, (int)diagonal); break;
        case 2: triu_kernel<__half><<<blocks, threads, 0, stream>>>(
                    (__half*)output, (const __half*)input, (int)num_elements, (int)rows, (int)cols, (int)diagonal); break;
        case 3: triu_kernel<uint8_t><<<blocks, threads, 0, stream>>>(
                    (uint8_t*)output, (const uint8_t*)input, (int)num_elements, (int)rows, (int)cols, (int)diagonal); break;
        case 4: triu_kernel<int><<<blocks, threads, 0, stream>>>(
                    (int*)output, (const int*)input, (int)num_elements, (int)rows, (int)cols, (int)diagonal); break;
        case 5: triu_kernel<int64_t><<<blocks, threads, 0, stream>>>(
                    (int64_t*)output, (const int64_t*)input, (int)num_elements, (int)rows, (int)cols, (int)diagonal); break;
        case 6: triu_kernel<bool><<<blocks, threads, 0, stream>>>(
                    (bool*)output, (const bool*)input, (int)num_elements, (int)rows, (int)cols, (int)diagonal); break;
    }
}
