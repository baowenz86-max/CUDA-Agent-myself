#include <cuda_runtime.h>
#include <math.h>

// ---------------- Upsample bilinear 2x (NCHW, float32, contiguous) ----------------
// 参考 PyTorch align_corners=False 的坐标映射：
//   src_x = (dst_x + 0.5) / scale - 0.5 = (dst_x + 0.5) * 0.5 - 0.5
//   对源坐标先 floor 再线性插值，边界 clamp 到 [0, W-1] / [0, H-1]
// 输入输出均为连续 NCHW，步长由形状决定。
// 线程映射：每个线程负责一个输出像素 (w, h)，block 维度为 (32, 8)，grid 维度覆盖整个输出平面。

template <typename T>
__global__ void upsample_bilinear2x_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int plane = blockIdx.z;

    if (plane >= N * C) return;
    if (idx >= outW || idy >= outH) return;

    int n = plane / C;
    int c = plane - n * C;

    // 连续 NCHW 下，输入基地址偏移
    const T* in_plane = input + (size_t)n * C * H * W + (size_t)c * H * W;
    T* out_plane = output + (size_t)n * C * outH * outW + (size_t)c * outH * outW;

    int dst_x = idx;
    int dst_y = idy;

    // 使用 double 提高坐标计算精度，避免边界误差
    double src_x_f = (static_cast<double>(dst_x) + 0.5) * 0.5 - 0.5;
    double src_y_f = (static_cast<double>(dst_y) + 0.5) * 0.5 - 0.5;

    // 处理边界：clamp 到 [0, W-1] 和 [0, H-1]
    if (src_x_f < 0.0) src_x_f = 0.0;
    if (src_y_f < 0.0) src_y_f = 0.0;
    int W_minus_1 = W - 1;
    int H_minus_1 = H - 1;
    if (src_x_f > W_minus_1) src_x_f = W_minus_1;
    if (src_y_f > H_minus_1) src_y_f = H_minus_1;

    int x0 = static_cast<int>(src_x_f);
    int y0 = static_cast<int>(src_y_f);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    if (x1 > W_minus_1) x1 = W_minus_1;
    if (y1 > H_minus_1) y1 = H_minus_1;

    double wx1 = src_x_f - x0;
    double wy1 = src_y_f - y0;
    double wx0 = 1.0 - wx1;
    double wy0 = 1.0 - wy1;

    // 读取四个源像素（连续内存）
    T v00 = in_plane[y0 * W + x0];
    T v01 = in_plane[y0 * W + x1];
    T v10 = in_plane[y1 * W + x0];
    T v11 = in_plane[y1 * W + x1];

    double result = wx0 * (wy0 * static_cast<double>(v00) + wy1 * static_cast<double>(v10)) +
                    wx1 * (wy0 * static_cast<double>(v01) + wy1 * static_cast<double>(v11));

    out_plane[dst_y * outW + dst_x] = static_cast<T>(result);
}

// ---------------- Ceil kernel ----------------
// 逐元素 ceil，输入输出均为连续内存。
// 使用一维 grid-stride 循环，每个线程处理多个元素。
template <typename T>
__global__ void ceil_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    long long numel
) {
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (; idx < numel; idx += stride) {
        output[idx] = static_cast<T>(ceilf(static_cast<float>(input[idx])));
    }
}

// ---------------- Launchers (extern C) ----------------

extern "C" void upsample_bilinear2x_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    cudaStream_t stream
) {
    dim3 block(32, 8);
    dim3 grid((outW + block.x - 1) / block.x,
              (outH + block.y - 1) / block.y,
              N * C);

    upsample_bilinear2x_kernel<float><<<grid, block, 0, stream>>>(
        input, output, N, C, H, W, outH, outW);
}

extern "C" void ceil_launcher(
    const float* input,
    float* output,
    long long numel,
    cudaStream_t stream
) {
    const int block_size = 256;
    long long total_blocks = (numel + block_size - 1) / block_size;
    int grid = static_cast<int>(total_blocks);
    ceil_kernel<float><<<grid, block_size, 0, stream>>>(input, output, numel);
}
