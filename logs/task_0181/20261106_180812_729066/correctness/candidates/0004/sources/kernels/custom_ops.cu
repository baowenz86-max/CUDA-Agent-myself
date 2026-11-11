#include <cuda_runtime.h>
#include <stdint.h>

// Helper to get next power of two
__device__ int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

// -----------------------------------------------------------------------------
// Argsort along last dimension using bitonic sort per row
// Block handles one row. Shared memory holds padded keys and indices.
// -----------------------------------------------------------------------------
__global__ void argsort_bitonic_kernel(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int outer,
    int seq_len,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    int batch_idx,        // batch index for this row
    int channel_idx)      // channel index for this row
{
    extern __shared__ char smem[];
    float* keys = reinterpret_cast<float*>(smem);
    int64_t* indices = reinterpret_cast<int64_t*>(keys + next_pow2(seq_len)); // offset by padded size of floats

    int idx = blockIdx.x; // row index (0..outer-1)
    if (idx >= outer) return;

    // Compute row pointers using provided batch/channel indices
    const float* in_row = input + batch_idx * in_stride_batch + channel_idx * in_stride_channel;
    int64_t* out_row = output + batch_idx * out_stride_batch + channel_idx * out_stride_channel;

    int padded = next_pow2(seq_len);

    // Load data into shared memory, pad with +infinity and dummy index
    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        if (i < seq_len) {
            keys[i] = in_row[i * in_stride_seq];
            indices[i] = i;
        } else {
            keys[i] = __int_as_float(0x7f800000); // +inf
            indices[i] = seq_len; // dummy
        }
    }
    __syncthreads();

    // Bitonic sort with stable tie-breaking using original index
    for (int k = 2; k <= padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < padded / 2; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    float key_i = keys[i];
                    float key_ixj = keys[ixj];
                    int64_t idx_i = indices[i];
                    int64_t idx_ixj = indices[ixj];
                    bool swap = false;
                    if (ascending) {
                        if (key_i > key_ixj || (key_i == key_ixj && idx_i > idx_ixj)) swap = true;
                    } else {
                        if (key_i < key_ixj || (key_i == key_ixj && idx_i > idx_ixj)) swap = true;
                    }
                    if (swap) {
                        keys[i] = key_ixj;
                        keys[ixj] = key_i;
                        indices[i] = idx_ixj;
                        indices[ixj] = idx_i;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Write back first seq_len indices
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        out_row[i * out_stride_seq] = indices[i];
    }
}

extern "C" void argsort_forward_launcher(
    const float* input,
    int64_t* output,
    int batch,
    int channels,
    int seq_len,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    cudaStream_t stream)
{
    int outer = batch * channels;
    int padded = next_pow2(seq_len); // may be called from host? No, need host version; compute manually.
    // Host-side computation:
    int padded_host = 1;
    while (padded_host < seq_len) padded_host <<= 1;

    int threads = (padded_host > 1024) ? 1024 : padded_host;
    if (threads < 32) threads = 32;
    threads = (threads + 31) / 32 * 32;

    size_t smem_size = padded_host * (sizeof(float) + sizeof(int64_t));
    cudaFuncSetAttribute(argsort_bitonic_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    // Launch one block per row; pass batch_idx and channel_idx for each row
    for (int row = 0; row < outer; ++row) {
        int b = row / channels;
        int c = row % channels;
        argsort_bitonic_kernel<<<1, threads, smem_size, stream>>>(
            input, output, outer, seq_len,
            in_stride_batch, in_stride_channel, in_stride_seq,
            out_stride_batch, out_stride_channel, out_stride_seq,
            b, c);
        // Note: Launching one block per row in a loop is inefficient but correct.
        // For performance, we should use a grid of outer blocks, but rolling loop is simpler.
        // We'll use a grid of outer blocks instead to avoid host loop:
        // Actually we can't pass different b,c per block if we launch a grid; we'd need to compute b,c from blockIdx.x,
        // which would require channels parameter. We can instead pass channels as a kernel argument and compute b,c inside.
        // To avoid undefined channels, we can add channels to kernel signature. Let's do that for efficiency.
    }
}

// More efficient launcher using grid of outer blocks:
extern "C" void argsort_forward_launcher(
    const float* input,
    int64_t* output,
    int batch,
    int channels,
    int seq_len,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    cudaStream_t stream)
{
    // Overload not allowed; we'll replace the previous definition entirely.
    // The correct implementation is below.
}

// Actual launcher (remove duplicate from above; keep only one)
// We'll implement using grid of blocks, and modify kernel to compute b,c from blockIdx.x and channels.
// To avoid undefined channels, we add 'int channels' to kernel.
__global__ void argsort_bitonic_kernel_v2(
    const float* __restrict__ input,
    int64_t* __restrict__ output,
    int outer,
    int seq_len,
    int channels,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq)
{
    extern __shared__ char smem[];
    int padded = next_pow2(seq_len);
    float* keys = reinterpret_cast<float*>(smem);
    int64_t* indices = reinterpret_cast<int64_t*>(keys + padded);

    int idx = blockIdx.x;
    if (idx >= outer) return;

    int b = idx / channels;
    int c = idx % channels;
    const float* in_row = input + b * in_stride_batch + c * in_stride_channel;
    int64_t* out_row = output + b * out_stride_batch + c * out_stride_channel;

    for (int i = threadIdx.x; i < padded; i += blockDim.x) {
        if (i < seq_len) {
            keys[i] = in_row[i * in_stride_seq];
            indices[i] = i;
        } else {
            keys[i] = __int_as_float(0x7f800000);
            indices[i] = seq_len;
        }
    }
    __syncthreads();

    for (int k = 2; k <= padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < padded / 2; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    float key_i = keys[i];
                    float key_ixj = keys[ixj];
                    int64_t idx_i = indices[i];
                    int64_t idx_ixj = indices[ixj];
                    bool swap = false;
                    if (ascending) {
                        if (key_i > key_ixj || (key_i == key_ixj && idx_i > idx_ixj)) swap = true;
                    } else {
                        if (key_i < key_ixj || (key_i == key_ixj && idx_i > idx_ixj)) swap = true;
                    }
                    if (swap) {
                        keys[i] = key_ixj;
                        keys[ixj] = key_i;
                        indices[i] = idx_ixj;
                        indices[ixj] = idx_i;
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        out_row[i * out_stride_seq] = indices[i];
    }
}

extern "C" void argsort_forward_launcher(
    const float* input,
    int64_t* output,
    int batch,
    int channels,
    int seq_len,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    cudaStream_t stream)
{
    int outer = batch * channels;
    int padded_host = 1;
    while (padded_host < seq_len) padded_host <<= 1;

    int threads = (padded_host > 1024) ? 1024 : padded_host;
    if (threads < 32) threads = 32;
    threads = (threads + 31) / 32 * 32;

    size_t smem_size = padded_host * (sizeof(float) + sizeof(int64_t));
    cudaFuncSetAttribute(argsort_bitonic_kernel_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

    argsort_bitonic_kernel_v2<<<outer, threads, smem_size, stream>>>(
        input, output, outer, seq_len, channels,
        in_stride_batch, in_stride_channel, in_stride_seq,
        out_stride_batch, out_stride_channel, out_stride_seq);
}

// -----------------------------------------------------------------------------
// Constant pad1d
// -----------------------------------------------------------------------------
__global__ void constant_pad1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int batch,
    int channels,
    int in_seq_len,
    int out_seq_len,
    int pad_left,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    float value)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_out = batch * channels * out_seq_len;
    if (idx >= total_out) return;

    int out_pos = idx % out_seq_len;
    int out_row = idx / out_seq_len;
    int b = out_row / channels;
    int c = out_row % channels;

    int64_t in_offset = b * in_stride_batch + c * in_stride_channel;
    int64_t out_offset = b * out_stride_batch + c * out_stride_channel;

    int in_pos = out_pos - pad_left;
    if (in_pos >= 0 && in_pos < in_seq_len) {
        output[out_offset + out_pos * out_stride_seq] = input[in_offset + in_pos * in_stride_seq];
    } else {
        output[out_offset + out_pos * out_stride_seq] = value;
    }
}

extern "C" void constant_pad1d_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int in_seq_len,
    int out_seq_len,
    int pad_left,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    float value,
    cudaStream_t stream)
{
    int total_out = batch * channels * out_seq_len;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    constant_pad1d_kernel<<<blocks, threads, 0, stream>>>(
        input, output, batch, channels, in_seq_len, out_seq_len, pad_left,
        in_stride_batch, in_stride_channel, in_stride_seq,
        out_stride_batch, out_stride_channel, out_stride_seq, value);
}

// -----------------------------------------------------------------------------
// Log softmax along last dimension
// -----------------------------------------------------------------------------
__global__ void log_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int outer,
    int seq_len,
    int channels,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq)
{
    extern __shared__ char smem[];
    float* s_max = reinterpret_cast<float*>(smem);
    float* s_sum = reinterpret_cast<float*>(smem + blockDim.x * sizeof(float));

    int row = blockIdx.x;
    if (row >= outer) return;

    int b = row / channels;
    int c = row % channels;
    const float* in_row = input + b * in_stride_batch + c * in_stride_channel;
    float* out_row = output + b * out_stride_batch + c * out_stride_channel;

    int tid = threadIdx.x;
    float local_max = -INFINITY;
    float local_sum = 0.0f;

    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = in_row[i * in_stride_seq];
        if (val > local_max) local_max = val;
    }
    s_max[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_max[tid + stride] > s_max[tid]) s_max[tid] = s_max[tid + stride];
        }
        __syncthreads();
    }
    float row_max = s_max[0];

    for (int i = tid; i < seq_len; i += blockDim.x) {
        float val = in_row[i * in_stride_seq];
        local_sum += __expf(val - row_max);
    }
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = s_sum[0];
    float log_sum = __logf(row_sum);

    for (int i = tid; i < seq_len; i += blockDim.x) {
        out_row[i * out_stride_seq] = in_row[i * in_stride_seq] - row_max - log_sum;
    }
}

extern "C" void log_softmax_forward_launcher(
    const float* input,
    float* output,
    int batch,
    int channels,
    int seq_len,
    int64_t in_stride_batch,
    int64_t in_stride_channel,
    int64_t in_stride_seq,
    int64_t out_stride_batch,
    int64_t out_stride_channel,
    int64_t out_stride_seq,
    cudaStream_t stream)
{
    int outer = batch * channels;
    int threads = 256;
    size_t smem_size = 2 * threads * sizeof(float);
    log_softmax_kernel<<<outer, threads, smem_size, stream>>>(
        input, output, outer, seq_len, channels,
        in_stride_batch, in_stride_channel, in_stride_seq,
        out_stride_batch, out_stride_channel, out_stride_seq);
}
