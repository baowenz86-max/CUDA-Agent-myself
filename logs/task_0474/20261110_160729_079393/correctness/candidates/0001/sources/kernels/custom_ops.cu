// CUDA kernels for AdaptiveLogSoftmaxWithLoss, softmax, clone
#include <cuda_runtime.h>

// Define block size for row-wise operations
#define ROW_BLOCK_SIZE 256

// Helper: matrix-vector product for head/tail linear layers.
// Each thread computes one output element of a row (e.g., y[row, col] = sum_k weight[col, k] * x[row, k])
// Since we use a per-row thread block, each block handles one row. 
// To improve occupancy, we could handle multiple rows per block, but for simplicity one block per row with COL_LOOP.
// grid = batch_size, block = min(out_features, 256) with grid-stride loop over out_features.
template<int TILE_DIM>
__global__ void matvec_kernel(const float* weight, const float* x, float* y, 
                              int batch_size, int in_features, int out_features) {
    int row = blockIdx.x;
    if (row >= batch_size) return;
    for (int col = threadIdx.x; col < out_features; col += blockDim.x) {
        float sum = 0.0f;
        const float* w_row = weight + (size_t)col * in_features;
        const float* x_row = x + (size_t)row * in_features;
        for (int k = 0; k < in_features; ++k) {
            sum += w_row[k] * x_row[k];
        }
        y[(size_t)row * out_features + col] = sum;
    }
}

// ReLU activation on a matrix (tail hidden)
__global__ void relu_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        data[idx] = data[idx] > 0.0f ? data[idx] : 0.0f;
    }
}

// Row-wise log_softmax
__global__ void row_log_softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* in_row = input + (size_t)row * cols;
    float* out_row = output + (size_t)row * cols;

    // Find max for numerical stability
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = in_row[i];
        if (v > max_val) max_val = v;
    }
    // block reduction for max (could use shared memory, but loop over cols per thread is okay)
    __shared__ float smax[ROW_BLOCK_SIZE];
    smax[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        }
        __syncthreads();
    }
    max_val = smax[0];
    __syncthreads();

    // Compute exp sum
    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += __expf(in_row[i] - max_val);
    }
    __shared__ float ssum[ROW_BLOCK_SIZE];
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            ssum[threadIdx.x] += ssum[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = ssum[0];
    float log_sum = logf(sum);

    // output = input - max - log_sum
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = (in_row[i] - max_val) - log_sum;
    }
}

// Row-wise softmax (standard)
__global__ void row_softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* in_row = input + (size_t)row * cols;
    float* out_row = output + (size_t)row * cols;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = in_row[i];
        if (v > max_val) max_val = v;
    }
    __shared__ float smax[ROW_BLOCK_SIZE];
    smax[threadIdx.x] = max_val;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]);
        __syncthreads();
    }
    max_val = smax[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sum += __expf(in_row[i] - max_val);
    }
    __shared__ float ssum[ROW_BLOCK_SIZE];
    ssum[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s];
        __syncthreads();
    }
    sum = ssum[0];
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = __expf(in_row[i] - max_val) / sum;
    }
}

// Clone kernel (elementwise copy)
__global__ void clone_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        output[idx] = input[idx];
    }
}

// Combined adaptive log softmax launcher
// All tensors are assumed contiguous float32 on CUDA.
// cutoffs is passed as int64_t (here just one cutoff, but we treat it as head_size_minus_one? 
// Actually cutoffs[0]=5000, head output size = 5001.
// In the contract, cutoffs is int64_t representing the list value, but since only one cutoff, we pass the cutoff value directly.
// However model_new.py passes self.cutoffs as a list [5000]. The binding should convert list to int64_t (first element) or we accept int64_t here.
// We'll accept int64_t cutoff (head_size = cutoff + 1).
// div_value is double.
extern "C" void adaptive_log_softmax_launcher(
    const float* x,
    const float* head_weight,
    const float* tail_weight0,
    const float* tail_weight1,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t head_size,       // = cutoff + 1, e.g., 5001
    int64_t tail_hidden_size, // 192
    int64_t tail_size,        // 5000
    int64_t n_classes,        // 10000
    double div_value,
    cudaStream_t stream)
{
    // We need temporary buffers for head and tail outputs.
    // Since kernel launches are asynchronous, we allocate device memory using cudaMalloc.
    // Better: allocate via cudaMallocAsync or traditional cudaMalloc.
    // We assume valid pointers and sizes.
    float* head_out = nullptr;
    float* tail_h = nullptr;
    float* tail_out = nullptr;
    cudaMalloc(&head_out, batch_size * head_size * sizeof(float));
    cudaMalloc(&tail_h, batch_size * tail_hidden_size * sizeof(float));
    cudaMalloc(&tail_out, batch_size * tail_size * sizeof(float));
    if (!head_out || !tail_h || !tail_out) {
        // handle error later
        if (head_out) cudaFree(head_out);
        if (tail_h) cudaFree(tail_h);
        if (tail_out) cudaFree(tail_out);
        return;
    }

    // Launch head matvec: output shape [batch, head_size]
    int head_block_size = 256;
    if (head_size < head_block_size) head_block_size = head_size;
    matvec_kernel<256><<<batch_size, head_block_size, 0, stream>>>(
        head_weight, x, head_out, (int)batch_size, (int)in_features, (int)head_size);

    // Launch tail0 matvec: output shape [batch, tail_hidden_size]
    int tail_hidden_block_size = 256;
    if (tail_hidden_size < tail_hidden_block_size) tail_hidden_block_size = tail_hidden_size;
    matvec_kernel<256><<<batch_size, tail_hidden_block_size, 0, stream>>>(
        tail_weight0, x, tail_h, (int)batch_size, (int)in_features, (int)tail_hidden_size);

    // ReLU on tail_h
    int total_tail_h = batch_size * tail_hidden_size;
    int threads = 256;
    int blocks = (total_tail_h + threads - 1) / threads;
    relu_kernel<<<blocks, threads, 0, stream>>>(tail_h, total_tail_h);

    // Launch tail1 matvec: output shape [batch, tail_size]
    int tail_block_size = 256;
    if (tail_size < tail_block_size) tail_block_size = tail_size;
    matvec_kernel<256><<<batch_size, tail_block_size, 0, stream>>>(
        tail_weight1, tail_h, tail_out, (int)batch_size, (int)tail_hidden_size, (int)tail_size);

    // Apply log_softmax to head_out (head_size columns)
    row_log_softmax_kernel<<<batch_size, head_block_size, 0, stream>>>(
        head_out, head_out, (int)batch_size, (int)head_size);

    // Apply log_softmax to tail_out (tail_size columns)
    row_log_softmax_kernel<<<batch_size, tail_block_size, 0, stream>>>(
        tail_out, tail_out, (int)batch_size, (int)tail_size);

    // Scale tail_out by div_value and concatenate into output.
    // Output layout: first head_size columns from head_out, next tail_size from tail_out * div_value.
    // We can do this with a kernel that copies both parts.
    // One block per row, threads loop over total columns.
    {
        int total_cols = n_classes;
        int threads2 = 256;
        dim3 grid(batch_size);
        dim3 block(threads2);
        // We'll define a kernel inline? Better define a __global__ function.
        // We'll define copy_scaled_concat_kernel below.
        // Since we can't define inside function, we'll add a separate kernel.
    }

    // Define a kernel to copy head part and scaled tail part.
    // We'll add it above and call here.
    // For simplicity, we'll call it.
    extern __global__ void copy_scaled_concat_kernel(const float* head, const float* tail, float* out, int batch, int head_size, int tail_size, int total_size, double scale);
    copy_scaled_concat_kernel<<<batch_size, 256, 0, stream>>>(head_out, tail_out, output, (int)batch_size, (int)head_size, (int)tail_size, (int)n_classes, div_value);

    // Free temporaries (but careful: kernels are async, so we need to free after stream sync? Better use cudaFree after sync? 
    // We could rely on cudaFree not blocking, but it's safe to free after all kernels are done, but that would require synchronization.
    // Alternatively, use cudaMallocAsync with stream ordered deallocation, but cudaMalloc is fine if we free after a cudaStreamSynchronize? too slow.
    // We can use cudaFree and it will not free until kernels using it complete? Actually cudaFree does not synchronize; it frees immediately and can cause issues if kernels still running.
    // To be safe, we'll use cudaMallocAsync (if available) or allocate via torch in binding? But contract says binding should allocate? However .cu cannot use torch.
    // We'll implement a workaround: allocate temporary buffers in binding? But binding cannot allocate inside .cu per rules? Actually binding.cpp is separate and can use torch. But our launcher expects pointers to temporary buffers? Not realistic.
    // Since we cannot include torch allocator in .cu, we must manage device memory. We'll use cudaMalloc and cudaFree with a synchronization after all kernels? That adds overhead but may be acceptable for correctness first. Later optimization can use persistent workspace.
    // For now, we'll insert cudaStreamSynchronize(stream) before cudaFree to ensure kernels done. This is not optimal but safe.
    cudaStreamSynchronize(stream);
    if (head_out) cudaFree(head_out);
    if (tail_h) cudaFree(tail_h);
    if (tail_out) cudaFree(tail_out);
}

// Kernel to copy head and scaled tail into output
__global__ void copy_scaled_concat_kernel(const float* head, const float* tail, float* out, int batch, int head_size, int tail_size, int total_size, double scale) {
    int row = blockIdx.x;
    if (row >= batch) return;
    for (int idx = threadIdx.x; idx < total_size; idx += blockDim.x) {
        if (idx < head_size) {
            out[(size_t)row * total_size + idx] = head[(size_t)row * head_size + idx];
        } else {
            int tail_idx = idx - head_size;
            out[(size_t)row * total_size + idx] = tail[(size_t)row * tail_size + tail_idx] * (float)scale;
        }
    }
}

// Launcher for custom_softmax (row-wise softmax)
extern "C" void custom_softmax_launcher(
    const float* input,
    float* output,
    int64_t rows,
    int64_t cols,
    int64_t dim,  // dim is -1 normally, unused since we assume last dim
    cudaStream_t stream)
{
    // For now, ignore dim and assume last dimension (since contract only uses -1).
    // Use one block per row with 256 threads.
    int block_size = ROW_BLOCK_SIZE;
    if (cols < block_size) block_size = cols;
    dim3 grid(rows);
    dim3 block(block_size);
    row_softmax_kernel<<<grid, block, 0, stream>>>(input, output, (int)rows, (int)cols);
}

// Launcher for custom_clone
extern "C" void custom_clone_launcher(
    const float* input,
    float* output,
    int64_t n,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // Avoid grid dimension overflow for huge tensors, use grid-stride loop.
    // n = 2048*10000 = 20,480,000, blocks = 80000, that's within 2^31-1 likely okay.
    // But we can cap blocks to 65535 and use stride.
    if (blocks > 65535) blocks = 65535;
    clone_kernel<<<blocks, threads, 0, stream>>>(input, output, n);
}
