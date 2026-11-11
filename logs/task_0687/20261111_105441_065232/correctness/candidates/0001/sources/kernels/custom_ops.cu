#include <cuda_runtime.h>
#include <cstdint>

// Kernel for max along dim=0 for 2D tensor of shape [rows, cols]
// Each block handles one column; blockIdx.x = col index
// Threads within block reduce over rows
__global__ void max_dim0_kernel(const float* __restrict__ input,
                                 float* __restrict__ values,
                                 int64_t* __restrict__ indices,
                                 int rows, int cols) {
    int col = blockIdx.x;
    if (col >= cols) return;

    // Use shared memory for reduction (optional but efficient)
    extern __shared__ float smem_values[];
    extern __shared__ int64_t smem_indices[];
    // Note: shared memory arrays are separate; need to partition manually
    // Actually we'll use separate allocations in dynamic shared memory pointer
    // For simplicity, use block reduction with shared memory for values and indices
    // We'll allocate shared memory as one chunk and split
    // Here we use two extern arrays which will be handled by compiler as two separate dynamic shared memory arrays if declared, but we can't have two extern unsized arrays easily.
    // Instead, declare a single shared memory buffer of size 2*blockDim.x elements and use pointer arithmetic.
    // We'll reinterpret: first half float values, second half int64_t indices (memory alignment considerations)
    
    // Need to handle dynamic shared memory manually
    // We'll allocate in the launcher: blockDim.x * sizeof(float) + blockDim.x * sizeof(int64_t)
    // Then use char* smem = (char*)extern shared memory
    // But for simplicity and correctness, skip shared memory; each thread does linear scan over subset of rows and then we can reduce via atomic or serial within warp. 
    // Given rows=8192, cols=128, we can assign one block per column with 128 threads, each thread handles rows/128 = 64 elements. Then do a reduction within block using shared memory.
    // Let's implement shared memory properly.

    extern __shared__ char smem_raw[];
    float* s_values = reinterpret_cast<float*>(smem_raw);
    int64_t* s_indices = reinterpret_cast<int64_t*>(smem_raw + blockDim.x * sizeof(float));

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Initialize local max
    float local_max = -INFINITY; // use -CUDART_INF_F maybe include math_constants? We'll use -3.402823466e+38F or -FLT_MAX
    int64_t local_idx = -1;

    for (int r = tid; r < rows; r += stride) {
        float val = input[r * cols + col];
        if (val > local_max || (val == local_max && r < local_idx)) {
            // torch.max returns first occurrence of maximum? Actually for ties, torch.max returns the index of the first maximum? Let's check: torch.max returns the index of the first maximum since it uses a stable comparison? Not sure. We'll mimic by using >= not > to get first? Typically, torch.max uses a reduction that may return first occurrence. We'll attempt to match by using > only, if equal keep previous (lower index).
            if (val > local_max) {
                local_max = val;
                local_idx = r;
            }
        }
    }

    s_values[tid] = local_max;
    s_indices[tid] = local_idx;
    __syncthreads();

    // Reduce within block using tree reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            float other_val = s_values[tid + offset];
            int64_t other_idx = s_indices[tid + offset];
            if (other_val > s_values[tid] || (other_val == s_values[tid] && other_idx < s_indices[tid])) {
                s_values[tid] = other_val;
                s_indices[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        values[col] = s_values[0];
        indices[col] = s_indices[0];
    }
}

// Kernel for cumsum along dim=0 for 2D tensor [rows, cols]
// Each block handles one column; blockIdx.x = col
// Thread 0 in block sequentially accumulates; others idle (low occupancy but correct)
// For better performance, use parallel prefix sum per column, but simplicity first.
__global__ void cumsum_dim0_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int rows, int cols) {
    int col = blockIdx.x;
    if (col >= cols) return;

    // single thread accumulates
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int r = 0; r < rows; ++r) {
            sum += input[r * cols + col];
            output[r * cols + col] = sum;
        }
    }
}

// Launcher functions with C linkage
extern "C" void max_dim0_launcher(const float* input,
                                  float* values,
                                  int64_t* indices,
                                  int rows,
                                  int cols,
                                  cudaStream_t stream) {
    int threads = 128; // good for reduction
    int blocks = cols;
    size_t shared_mem = threads * sizeof(float) + threads * sizeof(int64_t);
    max_dim0_kernel<<<blocks, threads, shared_mem, stream>>>(input, values, indices, rows, cols);
}

extern "C" void cumsum_dim0_launcher(const float* input,
                                     float* output,
                                     int rows,
                                     int cols,
                                     cudaStream_t stream) {
    int threads = 1; // one thread per block, block per column
    int blocks = cols;
    cumsum_dim0_kernel<<<blocks, threads, 0, stream>>>(input, output, rows, cols);
}
