#include <cuda_runtime.h>

// 逐元素绝对值 kernel
// 支持任意 stride，使用 grid-stride loop 覆盖所有元素
__global__ void abs_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    long long input_strides[],   // 最多支持 8 维，但本任务中固定 4 维，可使用指针数组
    long long output_strides[],
    int ndim
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long flat = idx; flat < numel; flat += stride) {
        // 将线性索引转换为多维坐标
        long long linear = flat;
        long long input_offset = 0;
        long long output_offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            long long coord = linear % (input_strides[d] / (d == 0 ? 1 : input_strides[d - 1])); // 注意：这个计算有误，需要实际 shape，但我们可以用另一种方法
        }
        // 由于上面的循环依赖 shape，而 launcher 没有传递 shape，这里采用更直接的方法：
        // 但实际上对于任意 stride，我们需要 shape。但我们可以利用输入和输出 stride 相同（本任务中输出 stride 与输入一致），
        // 然而 launcher 没有 shape，所以不能重构坐标。
        // 因此需要修改 launcher 传递 shape 或者使用另一种方式：直接对连续内存编译？
        // 根据契约，输出 stride 与输入一致，并且 shape 固定 [512,32,128,128]，但为了实现通用，
        // 我们可以通过传递 shape 解决。重新设计 launcher 参数。
        // 暂时在 kernel 中使用自定义的 shape 解码。
    }
}

// 实际上上面的实现有误，因为 numel 已知，我们只需维护一个从 flat 到多维索引的映射。
// 由于需要 shape，我们可以在 launcher 中计算好 shape 数组并传递给 kernel。
// 修改策略：launcher 接收 shape 指针。

// 重新实现更健壮的版本：
// 为了避免传递数组导致的复杂性，我们约定最多 8 维，使用固定大小数组作为 kernel 参数（通过值传递结构体）
// 但为了简单，这里直接定义两个 kernel 假设输入和输出都是连续的，因为尽管契约说 supports strides，
// 但实际输入来自 torch.randn 默认连续，输出 empty_like 也是连续。不过为了安全，我们可以显式处理 stride，
// 但在这里为了代码简洁，先实现连续版本，并在 binding 调用前确保连续？但 binding 不在本阶段。
// 根据 SKILL.md 要求，launcher 应支持任意 stride，但本阶段只写 .cu，binding 下一阶段会处理。
// 我们可以在 launcher 中将 data_ptr 视为连续，因为 binding 会保证 .contiguous()。
// 或者我们接受 stride 作为参数，但仅处理与 shape 一致的情况。

// 最终决定：以最通用方式实现，launcher 传递 shape、input_stride、output_stride 三个数组，kernel 内部处理任意 stride。
// 为了简单，我们仅支持 int64 的维度数据，并使用结构体传递。

// 定义结构体保存维度信息
struct StrideInfo {
    long long shape[8];
    long long strides[8];
    int ndim;
};

// 通用逐元素 kernel 模板
__global__ void abs_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    StrideInfo input_info,
    StrideInfo output_info
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long flat = idx; flat < numel; flat += stride) {
        // 将 flat 转换为多维坐标
        long long remaining = flat;
        long long input_offset = 0;
        long long output_offset = 0;
        for (int d = input_info.ndim - 1; d >= 0; --d) {
            long long coord = remaining % input_info.shape[d];
            remaining /= input_info.shape[d];
            input_offset += coord * input_info.strides[d];
            output_offset += coord * output_info.strides[d];
        }
        output[output_offset] = fabsf(input[input_offset]);
    }
}

// hardsigmoid kernel
__global__ void hardsigmoid_kernel(
    float* __restrict__ output,
    const float* __restrict__ input,
    long long numel,
    StrideInfo input_info,
    StrideInfo output_info
) {
    long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;

    for (long long flat = idx; flat < numel; flat += stride) {
        long long remaining = flat;
        long long input_offset = 0;
        long long output_offset = 0;
        for (int d = input_info.ndim - 1; d >= 0; --d) {
            long long coord = remaining % input_info.shape[d];
            remaining /= input_info.shape[d];
            input_offset += coord * input_info.strides[d];
            output_offset += coord * output_info.strides[d];
        }
        float val = input[input_offset];
        float hs = val / 6.0f + 0.5f;
        hs = fminf(fmaxf(hs, 0.0f), 1.0f);
        output[output_offset] = hs;
    }
}

// launcher: abs
// 参数：output, input, shape_data, input_strides_data, output_strides_data, ndim, numel, stream
// 注意：shape/strides 数据由 binding 从 Tensor 提取并传递为 long long 数组（在主机内存或设备内存？）
// 我们传递指针，launcher 将数据复制到 kernel 参数中的结构体（值传递），因此必须在主机端读取。
// 因此 launcher 接收 long long* (host pointers) 并填充 StrideInfo。
extern "C" void abs_forward_launcher(
    float* output,
    const float* input,
    const long long* shape,
    const long long* input_strides,
    const long long* output_strides,
    int ndim,
    long long numel,
    cudaStream_t stream
) {
    if (numel == 0) return;

    StrideInfo in_info, out_info;
    in_info.ndim = ndim;
    out_info.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        in_info.shape[i] = shape[i];
        in_info.strides[i] = input_strides[i];
        out_info.shape[i] = shape[i];
        out_info.strides[i] = output_strides[i];
    }

    int block = 256;
    long long grid_ll = (numel + block - 1) / block;
    // 限制 grid 大小，使用 grid-stride loop 防止溢出
    int max_grid = 65535 * 32; // 足够大
    if (grid_ll > max_grid) grid_ll = max_grid;
    dim3 grid((unsigned int)grid_ll);
    dim3 block_dim(block);

    abs_kernel<<<grid, block_dim, 0, stream>>>(
        output, input, numel, in_info, out_info);
}

// launcher: hardsigmoid
extern "C" void hardsigmoid_forward_launcher(
    float* output,
    const float* input,
    const long long* shape,
    const long long* input_strides,
    const long long* output_strides,
    int ndim,
    long long numel,
    cudaStream_t stream
) {
    if (numel == 0) return;

    StrideInfo in_info, out_info;
    in_info.ndim = ndim;
    out_info.ndim = ndim;
    for (int i = 0; i < ndim; ++i) {
        in_info.shape[i] = shape[i];
        in_info.strides[i] = input_strides[i];
        out_info.shape[i] = shape[i];
        out_info.strides[i] = output_strides[i];
    }

    int block = 256;
    long long grid_ll = (numel + block - 1) / block;
    int max_grid = 65535 * 32;
    if (grid_ll > max_grid) grid_ll = max_grid;
    dim3 grid((unsigned int)grid_ll);
    dim3 block_dim(block);

    hardsigmoid_kernel<<<grid, block_dim, 0, stream>>>(
        output, input, numel, in_info, out_info);
}
