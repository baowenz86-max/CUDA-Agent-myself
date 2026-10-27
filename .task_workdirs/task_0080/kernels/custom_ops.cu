#include <cuda_runtime.h>
#include <stdint.h>

// 前向声明 TensorMeta 结构体，在 launcher 和 kernel 中使用
struct TensorMeta {
    int ndim;
    int64_t shape[4];
    int64_t strides[4];
};

// ------------------------------------------------------------------------------
// Utility: check CUDA error after launches (host side, safe to call in launcher)
// ------------------------------------------------------------------------------
static inline void check_cuda_error_(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // 避免静默失败，binding 稍后抛出
    }
}

// 计算 mask 偏移：给定输入坐标（按广播后规则），mask_meta 为广播后的元数据
__device__ __forceinline__ int64_t compute_mask_offset(const int64_t* coords,
                                                       int input_ndim,
                                                       const TensorMeta& mask_meta) {
    int64_t offset = 0;
    int mask_dim = mask_meta.ndim - 1;
    for (int d = input_ndim - 1; d >= 0 && mask_dim >= 0; --d) {
        int64_t coord = coords[d];
        int64_t mask_extent = mask_meta.shape[mask_dim];
        int64_t idx = (mask_extent == 1) ? 0 : coord;
        offset += idx * mask_meta.strides[mask_dim];
        --mask_dim;
    }
    return offset;
}

// 计数内核：每个 block 处理一个连续 tile，统计选中的元素数
__global__ void count_selected_kernel(
    const float* __restrict__ input,
    const bool* __restrict__ mask,
    int64_t* __restrict__ block_counts,
    int64_t total_elements,
    TensorMeta input_meta,
    TensorMeta mask_meta)
{
    constexpr int ITEMS_PER_THREAD = 8;
    int64_t block_start = blockIdx.x * (int64_t)blockDim.x * ITEMS_PER_THREAD;
    int64_t block_end = min(block_start + (int64_t)blockDim.x * ITEMS_PER_THREAD, total_elements);
    int64_t local_count = 0;

    // 每个线程处理一些元素，按交错方式以提高内存合并
    for (int64_t base = block_start + threadIdx.x; base < block_end; base += blockDim.x) {
        int64_t idx = base;
        int64_t coords[4];
        for (int d = input_meta.ndim - 1; d >= 0; --d) {
            coords[d] = idx % input_meta.shape[d];
            idx /= input_meta.shape[d];
        }
        int64_t mask_offset = compute_mask_offset(coords, input_meta.ndim, mask_meta);
        if (mask[mask_offset]) {
            local_count++;
        }
    }

    // 块内归约
    __shared__ int64_t sdata[256];
    sdata[threadIdx.x] = local_count;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_counts[blockIdx.x] = sdata[0];
    }
}

// 块前缀和（单个线程循环）
__global__ void block_exclusive_scan_kernel(
    const int64_t* __restrict__ block_counts,
    int64_t* __restrict__ block_offsets,
    int64_t* __restrict__ total_count,
    int numBlocks)
{
    if (threadIdx.x == 0) {
        int64_t sum = 0;
        for (int i = 0; i < numBlocks; ++i) {
            block_offsets[i] = sum;
            sum += block_counts[i];
        }
        *total_count = sum;
    }
}

// 收集内核：根据块偏移写入选中元素
__global__ void gather_selected_kernel(
    const float* __restrict__ input,
    const bool* __restrict__ mask,
    float* __restrict__ output,
    const int64_t* __restrict__ block_offsets,
    int64_t total_elements,
    TensorMeta input_meta,
    TensorMeta mask_meta)
{
    constexpr int ITEMS_PER_THREAD = 8;
    int64_t block_start = blockIdx.x * (int64_t)blockDim.x * ITEMS_PER_THREAD;
    int64_t block_end = min(block_start + (int64_t)blockDim.x * ITEMS_PER_THREAD, total_elements);
    int64_t base_offset = block_offsets[blockIdx.x];

    // 每个线程负责连续子范围，以保证顺序稳定
    int64_t range_length = block_end - block_start;
    int64_t per_thread = (range_length + blockDim.x - 1) / blockDim.x;
    int64_t start = block_start + threadIdx.x * per_thread;
    int64_t end = min(start + per_thread, block_end);

    // 统计本线程选中数量
    int64_t local_count = 0;
    for (int64_t idx = start; idx < end; ++idx) {
        int64_t coords[4];
        int64_t tmp = idx;
        for (int d = input_meta.ndim - 1; d >= 0; --d) {
            coords[d] = tmp % input_meta.shape[d];
            tmp /= input_meta.shape[d];
        }
        int64_t mask_offset = compute_mask_offset(coords, input_meta.ndim, mask_meta);
        if (mask[mask_offset]) {
            local_count++;
        }
    }

    // 块内前缀和（Hillis-Steele）
    __shared__ int64_t s_counts[256];
    __shared__ int64_t s_prefix[256];
    s_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        int64_t temp = 0;
        if (threadIdx.x >= offset) {
            temp = s_counts[threadIdx.x - offset];
        }
        __syncthreads();
        if (threadIdx.x >= offset) {
            s_counts[threadIdx.x] += temp;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        s_prefix[0] = 0;
    } else {
        s_prefix[threadIdx.x] = s_counts[threadIdx.x - 1];
    }
    __syncthreads();

    // 写入输出
    int64_t out_pos = base_offset + s_prefix[threadIdx.x];
    for (int64_t idx = start; idx < end; ++idx) {
        int64_t coords[4];
        int64_t tmp = idx;
        for (int d = input_meta.ndim - 1; d >= 0; --d) {
            coords[d] = tmp % input_meta.shape[d];
            tmp /= input_meta.shape[d];
        }
        int64_t mask_offset = compute_mask_offset(coords, input_meta.ndim, mask_meta);
        if (mask[mask_offset]) {
            output[out_pos++] = input[idx];
        }
    }
}

// PReLU kernel
__global__ void prelu_kernel(const float* __restrict__ input, float* __restrict__ output, float weight, int64_t numel) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (idx < numel) {
        float x = input[idx];
        output[idx] = (x > 0.0f) ? x : (x * weight);
    }
}

// PReLU launcher
extern "C" void prelu_forward_launcher(
    const float* input,
    float* output,
    float weight,
    int64_t numel,
    cudaStream_t stream)
{
    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    prelu_kernel<<<blocks, threads, 0, stream>>>(input, output, weight, numel);
    check_cuda_error_("prelu_forward_launcher");
}

// masked_select 的两个阶段 launcher
extern "C" void masked_select_count_launcher(
    const float* input,
    const bool* mask,
    int64_t total_elements,
    TensorMeta input_meta,
    TensorMeta mask_meta,
    int64_t* block_counts,
    int64_t* block_offsets,
    int64_t* total_count,
    int num_blocks,
    cudaStream_t stream)
{
    int threads = 256;
    count_selected_kernel<<<num_blocks, threads, 0, stream>>>(
        input, mask, block_counts, total_elements, input_meta, mask_meta);
    check_cuda_error_("count_selected_kernel");

    block_exclusive_scan_kernel<<<1, 32, 0, stream>>>(
        block_counts, block_offsets, total_count, num_blocks);
    check_cuda_error_("block_exclusive_scan_kernel");
}

extern "C" void masked_select_gather_launcher(
    const float* input,
    const bool* mask,
    float* output,
    int64_t total_elements,
    TensorMeta input_meta,
    TensorMeta mask_meta,
    const int64_t* block_offsets,
    int num_blocks,
    cudaStream_t stream)
{
    int threads = 256;
    gather_selected_kernel<<<num_blocks, threads, 0, stream>>>(
        input, mask, output, block_offsets, total_elements, input_meta, mask_meta);
    check_cuda_error_("gather_selected_kernel");
}
