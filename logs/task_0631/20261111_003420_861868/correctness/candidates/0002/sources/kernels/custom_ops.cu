#include <cuda_runtime.h>
#include <cstdint>

//---------------------------------------------------------------------------
// sum_dim1_kernel: 对输入张量 x 的维度 1 求和，保持维度。
// 输入形状 [N, C, H, W]，输出形状 [N, 1, H, W]。
// 支持任意 stride，一个输出元素由一个线程完成。
//---------------------------------------------------------------------------
__global__ void sum_dim1_kernel_impl(const float* __restrict__ input,
                                     float* __restrict__ output,
                                     int64_t N, int64_t C, int64_t H, int64_t W,
                                     int64_t in_stride_n, int64_t in_stride_c,
                                     int64_t in_stride_h, int64_t in_stride_w,
                                     int64_t out_stride_n, int64_t out_stride_c,
                                     int64_t out_stride_h, int64_t out_stride_w) {
    int64_t total = N * H * W;
    int64_t global_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)gridDim.x * blockDim.x;

    for (int64_t idx = global_idx; idx < total; idx += stride) {
        int64_t n = idx / (H * W);
        int64_t rem = idx % (H * W);
        int64_t h = rem / W;
        int64_t w = rem % W;

        float sum = 0.0f;
        const float* base = input + n * in_stride_n + h * in_stride_h + w * in_stride_w;
        for (int64_t c = 0; c < C; ++c) {
            sum += base[c * in_stride_c];
        }

        float* out_ptr = output + n * out_stride_n + h * out_stride_h + w * out_stride_w;
        // 输出 shape [N,1,H,W]，out_stride_c 通常为 H*W，但由于 C=1，c=0 位置即可
        out_ptr[0] = sum;
    }
}

//---------------------------------------------------------------------------
// log10_kernel: 逐元素计算 log10(x)，支持任意 stride。
//---------------------------------------------------------------------------
__global__ void log10_kernel_impl(const float* __restrict__ input,
                                  float* __restrict__ output,
                                  int64_t numel) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        output[idx] = log10f(input[idx]);
    }
}

//---------------------------------------------------------------------------
// Launcher for sum_dim1_kernel
// 参数：input 指针，output 指针，各维度大小，各 stride，stream
//---------------------------------------------------------------------------
extern "C" void sum_dim1_kernel_launcher(const float* input,
                                        float* output,
                                        int64_t N, int64_t C, int64_t H, int64_t W,
                                        int64_t in_stride_n, int64_t in_stride_c,
                                        int64_t in_stride_h, int64_t in_stride_w,
                                        int64_t out_stride_n, int64_t out_stride_c,
                                        int64_t out_stride_h, int64_t out_stride_w,
                                        cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) {
        return; // 空张量，无操作
    }

    int64_t total = N * H * W;
    int threads = 256;
    int64_t blocks_64 = (total + threads - 1) / threads;
    // 限制 blocks 数量，使用 grid-stride loop
    if (blocks_64 > 65535) blocks_64 = 65535;
    int blocks = (int)blocks_64;

    sum_dim1_kernel_impl<<<blocks, threads, 0, stream>>>(
        input, output, N, C, H, W,
        in_stride_n, in_stride_c, in_stride_h, in_stride_w,
        out_stride_n, out_stride_c, out_stride_h, out_stride_w
    );
}

//---------------------------------------------------------------------------
// Launcher for log10_kernel
// 参数：input 指针，output 指针，元素总数，stream
//---------------------------------------------------------------------------
extern "C" void log10_kernel_launcher(const float* input,
                                     float* output,
                                     int64_t numel,
                                     cudaStream_t stream) {
    if (numel <= 0) {
        return;
    }

    int threads = 256;
    int64_t blocks_64 = (numel + threads - 1) / threads;
    if (blocks_64 > 65535) blocks_64 = 65535;
    int blocks = (int)blocks_64;

    log10_kernel_impl<<<blocks, threads, 0, stream>>>(input, output, numel);
}
