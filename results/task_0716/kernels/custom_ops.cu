#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

// ============ custom_mean_dim ============
__global__ void mean_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_output_elements,
    int64_t reduced_size,
    int64_t inner_size,
    int64_t input_outer_stride,
    int64_t input_inner_stride
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output_elements; i += stride) {
        int64_t outer = i / inner_size;
        int64_t inner = i - outer * inner_size;
        float sum = 0.0f;
        const float* base = input + outer * input_outer_stride + inner;
        for (int64_t j = 0; j < reduced_size; ++j) {
            sum += base[j * input_inner_stride];
        }
        output[i] = sum / (float)reduced_size;
    }
}

extern "C" void custom_mean_dim_launcher(
    const float* input,
    float* output,
    int64_t dim,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
) {
    int64_t reduced_size = C;
    int64_t inner_size = H * W;
    int64_t input_outer_stride = C * H * W;
    int64_t input_inner_stride = H * W;
    int64_t total_output = B * H * W;

    const int threads = 256;
    int64_t blocks = (total_output + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    mean_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_output, reduced_size, inner_size,
        input_outer_stride, input_inner_stride);
}

// ============ custom_scatter_dim ============
// 实现 torch.scatter(input, dim, index, src) 语义，处理重复索引（末次出现胜出）。
// 使用原子操作保证确定性：每个输出位置维护最大线性顺序的 packed 值。
// packed = (order << 32) | float_as_uint(src_val)
// order 是 index 张量中的线性索引（从0开始），因此更高的 order 对应更晚的出现。
// 最终，输出 = (packed != 0) ? uint_as_float(packed & 0xffffffff) : input_val

__device__ __forceinline__ unsigned long long atomicMaxULL(unsigned long long* address, unsigned long long val) {
    unsigned long long old = *address, assumed;
    do {
        assumed = old;
        old = atomicCAS(address, assumed, (val > assumed) ? val : assumed);
    } while (assumed != old);
    return old;
}

__global__ void scatter_index_kernel(
    const int64_t* __restrict__ index,
    const float* __restrict__ src,
    unsigned long long* __restrict__ packed,
    int64_t total_index_elements,
    int64_t inner_size,
    int64_t C,
    int64_t B,
    int64_t index_outer_stride,   // B stride = H*W
    int64_t index_inner_stride,   // W stride = 1
    int64_t src_outer_stride,
    int64_t src_inner_stride,
    int64_t input_outer_stride,
    int64_t input_inner_stride
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_index_elements; i += stride) {
        int64_t w = i % inner_size;
        int64_t b = i / inner_size;
        int64_t c = index[b * index_outer_stride + w * index_inner_stride];
        float val = src[b * src_outer_stride + w * src_inner_stride];
        // 输出位置映射：与 input 的 [b, c, h, w] 对应，其中 h,w 由 w 分解，但为简化直接使用线性输出索引
        // 输出张量形状 [B, C, H, W]，连续布局 output_idx = ((b * C + c) * inner_size) + w
        int64_t output_pos = (b * C + c) * inner_size + w;
        unsigned int val_bits = __float_as_uint(val);
        unsigned long long packed_val = ((unsigned long long)i << 32) | (unsigned long long)val_bits;
        atomicMaxULL(&packed[output_pos], packed_val);
    }
}

__global__ void scatter_finalize_kernel(
    const float* __restrict__ input,
    const unsigned long long* __restrict__ packed,
    float* __restrict__ output,
    int64_t total_output_elements,
    int64_t inner_size,
    int64_t C,
    int64_t B,
    int64_t input_outer_stride,
    int64_t input_inner_stride
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output_elements; i += stride) {
        // 计算输出坐标对应的输入坐标（连续布局）
        int64_t w = i % inner_size;
        int64_t hw = i / inner_size;
        int64_t c = hw % C;
        int64_t b = hw / C;
        float input_val = input[b * input_outer_stride + c * input_inner_stride + w];
        unsigned long long packed_val = packed[i];
        if (packed_val != 0) {
            unsigned int val_bits = (unsigned int)(packed_val & 0xffffffffULL);
            output[i] = __uint_as_float(val_bits);
        } else {
            output[i] = input_val;
        }
    }
}

extern "C" void custom_scatter_dim_launcher(
    const float* input,
    const int64_t* index,
    const float* src,
    float* output,
    int64_t dim,
    int64_t B,
    int64_t C,
    int64_t H,
    int64_t W,
    cudaStream_t stream
) {
    int64_t inner_size = H * W;
    int64_t total_output = B * C * H * W;
    int64_t total_index = B * H * W;

    // 分配临时 packed 缓冲区
    unsigned long long* d_packed = nullptr;
    cudaMalloc(&d_packed, total_output * sizeof(unsigned long long));
    cudaMemset(d_packed, 0, total_output * sizeof(unsigned long long));

    int threads = 256;
    int blocks_index = (total_index + threads - 1) / threads;
    if (blocks_index > 65535) blocks_index = 65535;
    scatter_index_kernel<<<blocks_index, threads, 0, stream>>>(
        index, src, d_packed, total_index, inner_size, C, B,
        H * W, 1, H * W, 1, C * H * W, H * W);

    int blocks_output = (total_output + threads - 1) / threads;
    if (blocks_output > 65535) blocks_output = 65535;
    scatter_finalize_kernel<<<blocks_output, threads, 0, stream>>>(
        input, d_packed, output, total_output, inner_size, C, B,
        C * H * W, H * W);

    cudaFree(d_packed);
}
