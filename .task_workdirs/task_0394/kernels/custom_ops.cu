#include <cuda_runtime.h>
#include <cstdint>

// Softmax along the innermost dimension (dim = 2) for a 3D tensor [batch, seq_len, dim_size]
// Each block processes one row of length dim_size.
__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ out,
                               int64_t batch,
                               int64_t seq_len,
                               int64_t dim_size) {
    int64_t row = blockIdx.y * static_cast<int64_t>(gridDim.x) + blockIdx.x; // flattened batch*seq_len index
    int64_t total_rows = batch * seq_len;
    if (row >= total_rows) return;

    const float* row_ptr = x + row * dim_size; // assuming contiguous layout
    float* out_ptr = out + row * dim_size;

    extern __shared__ float smem[]; // use shared memory for max and sum reduction
    float* s_max = smem;                     // size blockDim.x
    float* s_sum = smem + blockDim.x;        // size blockDim.x

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute max for numerical stability
    float local_max = -1e30f;
    for (int64_t i = tid; i < dim_size; i += stride) {
        float val = row_ptr[i];
        if (val > local_max) local_max = val;
    }
    s_max[tid] = local_max;
    __syncthreads();

    // Reduce max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_max[tid + s] > s_max[tid]) s_max[tid] = s_max[tid + s];
        }
        __syncthreads();
    }
    float row_max = s_max[0];

    // Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int64_t i = tid; i < dim_size; i += stride) {
        local_sum += expf(row_ptr[i] - row_max);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    // Reduce sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];

    // Avoid division by zero (should not happen for normal inputs but safe)
    float inv_sum = 1.0f / row_sum;

    // Write output
    for (int64_t i = tid; i < dim_size; i += stride) {
        out_ptr[i] = expf(row_ptr[i] - row_max) * inv_sum;
    }
}

__global__ void add_kernel(const float* __restrict__ x,
                           const float* __restrict__ bias,
                           float* __restrict__ out,
                           int64_t total_elements,
                           int64_t dim_size) {
    int64_t idx = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    int64_t stride = blockDim.x * static_cast<int64_t>(gridDim.x);
    for (int64_t i = idx; i < total_elements; i += stride) {
        int64_t bias_idx = i % dim_size;
        out[i] = x[i] + bias[bias_idx];
    }
}

extern "C" void custom_softmax_launcher(const float* x,
                                        float* out,
                                        int64_t batch,
                                        int64_t seq_len,
                                        int64_t dim_size,
                                        cudaStream_t stream) {
    int64_t total_rows = batch * seq_len;
    const int block_size = 256;
    // dim_size is 2048, so 256 threads is fine
    dim3 block(block_size, 1);
    dim3 grid(static_cast<unsigned int>(total_rows), 1);
    size_t shared_mem = 2 * block_size * sizeof(float);
    softmax_kernel<<<grid, block, shared_mem, stream>>>(x, out, batch, seq_len, dim_size);
}

extern "C" void custom_add_launcher(const float* x,
                                    const float* bias,
                                    float* out,
                                    int64_t total_elements,
                                    int64_t dim_size,
                                    cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    if (blocks > 65535) blocks = 65535; // grid dimension limit, grid-stride loop handles rest
    dim3 block(block_size);
    dim3 grid(static_cast<unsigned int>(blocks));
    add_kernel<<<grid, block, 0, stream>>>(x, bias, out, total_elements, dim_size);
}
