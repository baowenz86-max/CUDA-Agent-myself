// Complete CUDA kernels and launchers for custom ops
// No torch/ATen/pybind dependencies; only raw pointers, scalars, cudaStream_t

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

// -----------------------------------------------------------------------------
// Kernel: embedding_forward_kernel
// input_idx: int64 indices [batch_size, sequence_length] (GPU)
// weight: float32 [num_embeddings, embedding_dim] (GPU)
// output: float32 [batch_size, sequence_length, embedding_dim] (GPU)
// Each block handles one row, threads copy along embedding dim.
// -----------------------------------------------------------------------------
__global__ void embedding_forward_kernel(const int64_t* __restrict__ input_idx,
                                         const float* __restrict__ weight,
                                         float* __restrict__ output,
                                         int total_rows,
                                         int embedding_dim) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    int64_t idx = input_idx[row];
    const float* src = weight + idx * embedding_dim;
    float* dst = output + (size_t)row * embedding_dim;

    for (int d = threadIdx.x; d < embedding_dim; d += blockDim.x) {
        dst[d] = src[d];
    }
}

// Launcher: all pointers on GPU, no CPU copies.
extern "C" void embedding_forward_launcher(
    const int64_t* input_gpu,
    const float* weight_gpu,
    float* output_gpu,
    int total_rows,
    int embedding_dim,
    cudaStream_t stream) {

    // Use 256 threads per block, one block per row.
    int threads = 256;
    dim3 block(threads);
    dim3 grid(total_rows);
    embedding_forward_kernel<<<grid, block, 0, stream>>>(
        input_gpu, weight_gpu, output_gpu, total_rows, embedding_dim);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // The binding layer will check cudaGetLastError after launcher.
        // We can't throw here; just leave error in CUDA state.
    }
}

// -----------------------------------------------------------------------------
// Kernel: log_softmax_kernel
// input: [total_rows, D] (GPU), output same shape (GPU)
// Uses shared memory reduction, assuming D <= blockDim (512) and D power of two.
// -----------------------------------------------------------------------------
__global__ void log_softmax_kernel(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int total_rows,
                                   int D) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= total_rows) return;

    const float* in_row = input + (size_t)row * D;
    float* out_row = output + (size_t)row * D;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    // Load row into shared memory
    for (int i = tid; i < D; i += threads) {
        sdata[i] = in_row[i];
    }
    __syncthreads();

    // Find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < D; i += threads) {
        if (sdata[i] > max_val) max_val = sdata[i];
    }
    // Reduction for max in shared memory
    for (int offset = threads / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
    }
    __syncthreads();
    max_val = sdata[0];

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = tid; i < D; i += threads) {
        float e = __expf(sdata[i] - max_val);
        sdata[i] = e;
        sum += e;
    }
    // Reduction for sum
    for (int offset = threads / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            sdata[tid] = sdata[tid] + sdata[tid + offset];
        }
    }
    __syncthreads();
    float log_sum = logf(sdata[0]);

    // Write log_softmax = x - max - log_sum
    for (int i = tid; i < D; i += threads) {
        out_row[i] = in_row[i] - max_val - log_sum;
    }
}

extern "C" void log_softmax_forward_launcher(
    const float* input_gpu,
    float* output_gpu,
    int total_rows,
    int D,
    cudaStream_t stream) {

    int threads = 512;
    dim3 block(threads);
    dim3 grid(total_rows);
    size_t shared_bytes = D * sizeof(float);
    log_softmax_kernel<<<grid, block, shared_bytes, stream>>>(
        input_gpu, output_gpu, total_rows, D);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}

// -----------------------------------------------------------------------------
// Kernel: softmax_kernel
// -----------------------------------------------------------------------------
__global__ void softmax_kernel(const float* __restrict__ input,
                               float* __restrict__ output,
                               int total_rows,
                               int D) {
    extern __shared__ float sdata[];
    int row = blockIdx.x;
    if (row >= total_rows) return;

    const float* in_row = input + (size_t)row * D;
    float* out_row = output + (size_t)row * D;

    int tid = threadIdx.x;
    int threads = blockDim.x;

    for (int i = tid; i < D; i += threads) {
        sdata[i] = in_row[i];
    }
    __syncthreads();

    float max_val = -FLT_MAX;
    for (int i = tid; i < D; i += threads) {
        if (sdata[i] > max_val) max_val = sdata[i];
    }
    for (int offset = threads / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + offset]);
        }
    }
    __syncthreads();
    max_val = sdata[0];

    float sum = 0.0f;
    for (int i = tid; i < D; i += threads) {
        float e = __expf(sdata[i] - max_val);
        sdata[i] = e;
        sum += e;
    }
    for (int offset = threads / 2; offset > 0; offset >>= 1) {
        __syncthreads();
        if (tid < offset) {
            sdata[tid] = sdata[tid] + sdata[tid + offset];
        }
    }
    __syncthreads();
    float inv_sum = 1.0f / sdata[0];

    for (int i = tid; i < D; i += threads) {
        out_row[i] = sdata[i] * inv_sum;
    }
}

extern "C" void softmax_forward_launcher(
    const float* input_gpu,
    float* output_gpu,
    int total_rows,
    int D,
    cudaStream_t stream) {

    int threads = 512;
    dim3 block(threads);
    dim3 grid(total_rows);
    size_t shared_bytes = D * sizeof(float);
    softmax_kernel<<<grid, block, shared_bytes, stream>>>(
        input_gpu, output_gpu, total_rows, D);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}

// -----------------------------------------------------------------------------
// KL divergence elementwise and reduction
// -----------------------------------------------------------------------------
__global__ void kl_div_elementwise_kernel(const float* __restrict__ input,
                                         const float* __restrict__ target,
                                         float* __restrict__ output,
                                         int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float t = target[idx];
        float p = input[idx];
        float log_t = logf(t);
        output[idx] = t * (log_t - p);
    }
}

__global__ void kl_div_reduce_and_scale_kernel(const float* __restrict__ data,
                                               float* __restrict__ result,
                                               int N,
                                               float scale) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int threads = blockDim.x;

    float local_sum = 0.0f;
    for (int i = tid; i < N; i += threads) {
        local_sum += data[i];
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int offset = threads / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *result = sdata[0] * scale;
    }
}

extern "C" void kl_div_batchmean_forward_launcher_v2(
    const float* input_gpu,
    const float* target_gpu,
    float* output_gpu,
    int batch_size,
    int total_elements,
    cudaStream_t stream) {

    float* elementwise;
    cudaMallocAsync(&elementwise, (size_t)total_elements * sizeof(float), stream);

    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    kl_div_elementwise_kernel<<<blocks, threads, 0, stream>>>(
        input_gpu, target_gpu, elementwise, total_elements);

    int reduce_threads = 512;
    size_t shared_bytes = reduce_threads * sizeof(float);
    float scale = 1.0f / (float)batch_size;
    kl_div_reduce_and_scale_kernel<<<1, reduce_threads, shared_bytes, stream>>>(
        elementwise, output_gpu, total_elements, scale);

    cudaFreeAsync(elementwise, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // ignore
    }
}
