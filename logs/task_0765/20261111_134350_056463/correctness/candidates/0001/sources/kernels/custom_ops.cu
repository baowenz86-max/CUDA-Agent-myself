#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// custom_stack: 等价于 torch.stack([x1, x2], dim=0)
// 输入: x1, x2 均为 float32, 形状 [8192, 128], 连续
// 输出: out 形状 [2, 8192, 128], 连续
// -----------------------------------------------------------------------------
__global__ void custom_stack_kernel(float* __restrict__ out,
                                    const float* __restrict__ x1,
                                    const float* __restrict__ x2,
                                    int rows, int cols) {
    // total elements = 2 * rows * cols
    long long total = 2LL * rows * cols;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long i = idx; i < total; i += stride) {
        int batch = i / (rows * cols);           // 0 or 1
        int rem = i - batch * rows * cols;
        int row = rem / cols;
        int col = rem - row * cols;
        const float* src = (batch == 0) ? x1 : x2;
        out[i] = src[row * cols + col];
    }
}

extern "C" void custom_stack_launcher(float* out,
                                      const float* x1,
                                      const float* x2,
                                      int rows,
                                      int cols,
                                      cudaStream_t stream) {
    long long total = 2LL * rows * cols;
    int block = 256;
    long long grid_ll = (total + block - 1) / block;
    int grid = (int)grid_ll;
    custom_stack_kernel<<<grid, block, 0, stream>>>(out, x1, x2, rows, cols);
    cudaGetLastError();  // launcher 不抛出，错误由 binding 检查
}

// -----------------------------------------------------------------------------
// custom_bmm_transpose: 等价于 torch.bmm(stacked, stacked.transpose(1, 2))
// 输入: stacked 形状 [2, 8192, 128], 连续
// 输出: out 形状 [2, 8192, 8192], 连续
// 计算: out[b, i, j] = sum_k stacked[b, i, k] * stacked[b, j, k]
// -----------------------------------------------------------------------------
__global__ void custom_bmm_transpose_kernel(float* __restrict__ out,
                                            const float* __restrict__ stacked,
                                            int batch_size,
                                            int rows,
                                            int cols) {
    long long total = (long long)batch_size * rows * rows;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)blockDim.x * gridDim.x;

    for (long long linear = idx; linear < total; linear += stride) {
        int b = linear / (rows * rows);
        long long rem = linear - b * rows * rows;
        int i = rem / rows;
        int j = rem - i * rows;

        const float* base = stacked + b * rows * cols;
        float sum = 0.0f;
        // 小 K (128)，线程内串行点积，避免共享内存和数值顺序问题
        for (int k = 0; k < cols; ++k) {
            sum += base[i * cols + k] * base[j * cols + k];
        }
        out[linear] = sum;
    }
}

extern "C" void custom_bmm_transpose_launcher(float* out,
                                              const float* stacked,
                                              int batch_size,
                                              int rows,
                                              int cols,
                                              cudaStream_t stream) {
    long long total = (long long)batch_size * rows * rows;
    int block = 256;
    long long grid_ll = (total + block - 1) / block;
    int grid = (int)grid_ll;
    custom_bmm_transpose_kernel<<<grid, block, 0, stream>>>(out, stacked, batch_size, rows, cols);
    cudaGetLastError();
}
