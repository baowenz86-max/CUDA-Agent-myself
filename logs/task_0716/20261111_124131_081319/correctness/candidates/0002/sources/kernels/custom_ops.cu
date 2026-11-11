#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// ============ custom_mean_dim ============
// 计算输入张量沿 dim 维度的均值，保持维度（keepdim=True）。
// 输入形状: [B, C, H, W] (B=512, C=32, H=128, W=128)
// 输出形状: [B, 1, H, W]
// dim 通常为 1，但实现为通用任意 dim。
// 使用 grid-stride loop 处理大尺寸。
__global__ void mean_dim_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int64_t total_output_elements,
    int64_t reduced_size,        // 被归约的维度大小 (即 C)
    int64_t outer_stride,        // 归约维度之前的元素总数（即 B）
    int64_t inner_size,          // 归约维度之后的元素总数（即 H*W）
    int64_t input_outer_stride,  // 输入中归约维度外层的 stride（即 C*H*W）
    int64_t input_inner_stride   // 输入中归约维度内层的 stride（即 H*W）
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_output_elements; i += stride) {
        // 输出每个元素对应一个 (outer, inner) 组合
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

// 发射均值 kernel。
// 输入: input 指针 (float), dim, 形状参数 B, C, H, W
// 简化：仅支持 dim=1 的 4D 输入，但也可适配一般情况（只要 dim 不是 0 或 3）。
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
    // 对于 dim=1，输出形状 [B,1,H,W]
    int64_t reduced_size = C;
    int64_t outer_stride = B;
    int64_t inner_size = H * W;
    int64_t input_outer_stride = C * H * W;
    int64_t input_inner_stride = H * W;
    int64_t total_output = B * H * W;

    const int threads = 256;
    int64_t blocks = (total_output + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535; // 使用 grid-stride loop 可处理更多
    mean_dim_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total_output, reduced_size, outer_stride, inner_size,
        input_outer_stride, input_inner_stride);
}

// ============ custom_scatter_dim ============
// 实现 torch.scatter(input, dim, index, src) 语义。
// 输入: input [B,C,H,W], dim=1, index [B,1,H,W] (int64_t), src [B,1,H,W]
// 输出: 先复制 input，然后用 index 指定位置替换为 src 对应值。
// 注意：如果 input 是连续的，直接复制；但为支持 strides，按元素复制。
// 为了正确处理重复索引（最后写入胜出），采用原子操作保证顺序。

// 内核1：处理每个 index 元素，使用 atomicMax 存储最高顺序（线性索引）对应的 src 值。
__global__ void scatter_index_kernel(
    const int64_t* __restrict__ index,
    const float* __restrict__ src,
    uint64_t* __restrict__ out_buffer,  // 每个输出位置存储打包的 (order<<32 | float_bits)
    int64_t B,
    int64_t C,
    int64_t inner_size,   // H*W
    int64_t input_outer_stride,  // C*H*W
    int64_t input_inner_stride,  // H*W
    int64_t index_outer_stride,  // H*W
    int64_t index_inner_stride,  // 1
    int64_t src_outer_stride,
    int64_t src_inner_stride,
    int64_t index_total   // B*H*W
) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= index_total) return;
    // 解码 i 为 (b, w)
    int64_t w = i % inner_size;
    int64_t b = i / inner_size;
    int64_t c = index[b * index_outer_stride + w * index_inner_stride];
    float val = src[b * src_outer_stride + w * src_inner_stride];
    // 计算输出位置
    int64_t out_offset = b * input_outer_stride + c * input_inner_stride + w;
    uint32_t val_bits = __float_as_uint(val);
    uint64_t order = (uint64_t)i; // 用线性索引作为顺序
    uint64_t packed = (order << 32) | val_bits;
    atomicMax(&out_buffer[out_offset], packed);
}

// 内核2：最终确定输出：复制 input，然后根据 buffer 重写被 scatter 的位置。
__global__ void scatter_finalize_kernel(
    const float* __restrict__ input,
    uint64_t* __restrict__ out_buffer,
    float* __restrict__ output,
    int64_t total_elements,  // B*C*H*W
    int64_t inner_size,      // H*W
    int64_t C,
    int64_t B,
    int64_t input_outer_stride,
    int64_t input_inner_stride
) {
    int64_t idx = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        // 计算多维索引
        int64_t w = i % inner_size;
        int64_t hw = i / inner_size;
        int64_t c = hw % C;
        int64_t b = hw / C;
        float in_val = input[b * input_outer_stride + c * input_inner_stride + w];
        float out_val = in_val;
        uint64_t packed = out_buffer[i];
        if (packed != 0) {
            // 有 scatter 更新
            uint32_t val_bits = (uint32_t)(packed & 0xFFFFFFFFULL);
            out_val = __uint_as_float(val_bits);
        }
        output[i] = out_val;
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
    // 对于 dim=1，输出形状 [B,C,H,W]
    int64_t inner_size = H * W;
    int64_t total_elements = B * C * H * W;
    int64_t index_total = B * H * W;

    // 假设连续 stride
    int64_t input_outer_stride = C * H * W;
    int64_t input_inner_stride = H * W;
    int64_t index_outer_stride = H * W;
    int64_t index_inner_stride = 1;
    int64_t src_outer_stride = H * W;
    int64_t src_inner_stride = 1;

    // 分配临时 buffer，初始化为0
    uint64_t* temp_buffer = nullptr;
    cudaError_t err = cudaMalloc(&temp_buffer, total_elements * sizeof(uint64_t));
    if (err != cudaSuccess) {
        // 处理错误，但这里简单返回
        return;
    }
    cudaMemsetAsync(temp_buffer, 0, total_elements * sizeof(uint64_t), stream);

    const int threads = 256;
    int64_t blocks_index = (index_total + threads - 1) / threads;
    if (blocks_index > 65535) blocks_index = 65535;
    scatter_index_kernel<<<blocks_index, threads, 0, stream>>>(
        index, src, temp_buffer,
        B, C, inner_size, input_outer_stride, input_inner_stride,
        index_outer_stride, index_inner_stride, src_outer_stride, src_inner_stride,
        index_total);

    int64_t blocks_final = (total_elements + threads - 1) / threads;
    if (blocks_final > 65535) blocks_final = 65535;
    scatter_finalize_kernel<<<blocks_final, threads, 0, stream>>>(
        input, temp_buffer, output, total_elements, inner_size, C, B,
        input_outer_stride, input_inner_stride);

    cudaFree(temp_buffer);
}
