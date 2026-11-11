#include <cuda_runtime.h>

// 计算直方图：输入 float32 张量，统计 bins 个区间 [min + i*width, min + (i+1)*width)
// 最后一个区间包含 max，超出 [min, max] 的值忽略。
// 原子加累加，输出 float32 数组。
__global__ void histc_kernel(
    const float* __restrict__ input,
    int64_t total_elements,
    float* __restrict__ output,
    int64_t bins,
    float min_val,
    float max_val
) {
    // 计算 bin 宽度
    float width = (max_val - min_val) / static_cast<float>(bins);
    // 使用 grid-stride loop
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total_elements; i += stride) {
        float value = input[i];
        // 检查是否在 [min_val, max_val] 范围内
        if (value < min_val || value > max_val) {
            continue;
        }
        // 计算 bin 索引
        int64_t bin;
        if (value == max_val) {
            // 最后一个 bin 包含 max
            bin = bins - 1;
        } else {
            bin = static_cast<int64_t>((value - min_val) / width);
            // 数值容错，确保 bin 不超过 bins-1
            if (bin >= bins) {
                bin = bins - 1;
            }
            if (bin < 0) {
                bin = 0;
            }
        }
        atomicAdd(&output[bin], 1.0f);
    }
}

// 简单拷贝 kernel
__global__ void copy_kernel(const float* __restrict__ in, float* __restrict__ out, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        out[i] = in[i];
    }
}

// launcher: compute_histc
// 输入：input 指针，total_elements, bins, min, max, stream
// 输出：output 指针（需预先分配 bins 个 float 并初始化为 0）
extern "C" void compute_histc_launcher(
    const float* input,
    int64_t total_elements,
    float* output,
    int64_t bins,
    float min_val,
    float max_val,
    cudaStream_t stream
) {
    // 检查 bins 合法性
    if (bins <= 0 || total_elements <= 0) {
        return;
    }
    // 初始化 output 为 0（调用者可能传入未初始化的 tensor，需要先归零）
    // 这里用 cudaMemsetAsync 在 launcher 中完成
    cudaMemsetAsync(output, 0, bins * sizeof(float), stream);

    // 配置 kernel launch
    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    // 限制最大 blocks 数量，避免一次性启动过多
    const int max_blocks = 65535;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    // 启动 kernel（使用 grid-stride loop，所以 blocks 可以限制）
    histc_kernel<<<blocks, threads, 0, stream>>>(
        input,
        total_elements,
        output,
        bins,
        min_val,
        max_val
    );
}

// cat_histc 的 launcher：输入已经是扁平连续张量，语义为原样返回，
// 但绑定层可能调用此 launcher 进行复制（实现见 binding.cpp）。
// 此处提供空操作，binding 阶段可直接返回输入或调用复制。
extern "C" void cat_histc_launcher(
    const float* input,
    float* output,
    int64_t total_elements,
    cudaStream_t stream
) {
    // 如果 output != input，则进行拷贝
    if (input == output) {
        return;
    }
    int threads = 256;
    int blocks = static_cast<int>((total_elements + threads - 1) / threads);
    const int max_blocks = 65535;
    if (blocks > max_blocks) {
        blocks = max_blocks;
    }
    copy_kernel<<<blocks, threads, 0, stream>>>(input, output, total_elements);
}
