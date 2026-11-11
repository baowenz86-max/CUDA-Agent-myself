#include <cuda_runtime.h>
#include <cstdint>

// 逐元素符号函数 kernel。
// 输入输出均为 float32，支持非连续布局。
// 使用 grid-stride loop 保证覆盖所有元素，同时处理任意数量的元素。
__global__ void sign_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    int64_t numel,
    const int64_t* __restrict__ input_strides,
    const int64_t* __restrict__ output_strides,
    int ndim
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    for (int64_t linear = idx; linear < numel; linear += stride) {
        // 将线性索引转换为多维索引，并计算 input/output 的偏移量
        int64_t in_offset = 0;
        int64_t out_offset = 0;
        int64_t rem = linear;
        for (int d = ndim - 1; d >= 0; --d) {
            int64_t coord = rem % output_strides[d] == 0 ? 0 : 0; // 占位，实际需要 shape 信息
            // 注意：这里无法仅根据 stride 恢复 shape，需要额外传递 shape 或者采用另一种方式。
            // 我们改为传递 shape 数组。
        }
    }
}
