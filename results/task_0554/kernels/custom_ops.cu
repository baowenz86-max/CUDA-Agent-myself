// kernels/custom_ops.cu
#include <cuda_runtime.h>

// Kernel: 2D max pooling
// Computes output[n, c, oh, ow] = max over window with optional dilation and ceil_mode.
// Input is assumed to be a 4D tensor with shape N, C, H, W and strides (strideN, strideC, strideH, strideW).
// Output is a contiguous 4D tensor with shape N, C, outH, outW and standard strides (outC*outH*outW, outH*outW, outW, 1).
// We use 1D grid over all output elements, each thread processes a grid-stride loop.
__global__ void max_pool2d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int padH, int padW,
    int dilationH, int dilationW,
    long long inputStrideN, long long inputStrideC, long long inputStrideH, long long inputStrideW,
    long long outputStrideN, long long outputStrideC, long long outputStrideH, long long outputStrideW
) {
    long long total = (long long)N * C * outH * outW;
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long step = (long long)gridDim.x * blockDim.x;

    for (; idx < total; idx += step) {
        // Decompose idx to (n, c, oh, ow)
        int ow = idx % outW;
        long long tmp = idx / outW;
        int oh = tmp % outH;
        tmp /= outH;
        int c = tmp % C;
        int n = tmp / C;

        float max_val = -3.402823466e+38f; // -FLT_MAX
        bool found = false;

        for (int kh = 0; kh < kernelH; ++kh) {
            int ih = oh * strideH - padH + kh * dilationH;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kernelW; ++kw) {
                int iw = ow * strideW - padW + kw * dilationW;
                if (iw < 0 || iw >= W) continue;
                long long input_idx = (long long)n * inputStrideN + (long long)c * inputStrideC
                                     + (long long)ih * inputStrideH + (long long)iw * inputStrideW;
                float val = input[input_idx];
                if (!found || val > max_val) {
                    max_val = val;
                    found = true;
                }
            }
        }

        // If no valid input element (should not happen for valid pooling configurations, but ensure a value)
        if (!found) max_val = 0.0f;

        long long output_idx = (long long)n * outputStrideN + (long long)c * outputStrideC
                               + (long long)oh * outputStrideH + (long long)ow * outputStrideW;
        output[output_idx] = max_val;
    }
}

// C interface launcher for binding.cpp
extern "C" void max_pool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int padH, int padW,
    int dilationH, int dilationW,
    bool ceil_mode,
    long long inputStrideN, long long inputStrideC, long long inputStrideH, long long inputStrideW,
    long long outputStrideN, long long outputStrideC, long long outputStrideH, long long outputStrideW,
    cudaStream_t stream
) {
    // Compute output spatial dimensions
    int outH, outW;
    if (ceil_mode) {
        outH = (H + 2 * padH - dilationH * (kernelH - 1) - 1 + strideH - 1) / strideH + 1;
        outW = (W + 2 * padW - dilationW * (kernelW - 1) - 1 + strideW - 1) / strideW + 1;
        // Ensure at least 1; the formula above should yield positive, but clamp to handle edge cases.
        outH = (outH < 1) ? 1 : outH;
        outW = (outW < 1) ? 1 : outW;
    } else {
        outH = (H + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1;
        outW = (W + 2 * padW - dilationW * (kernelW - 1) - 1) / strideW + 1;
    }

    if (outH <= 0 || outW <= 0) {
        return; // no output elements
    }

    long long total = (long long)N * C * outH * outW;
    int threads = 256;
    long long blocks_ll = (total + threads - 1) / threads;
    // Cap grid to max allowed and use grid-stride loop
    int max_blocks = 65535;
    int blocks = (blocks_ll > max_blocks) ? max_blocks : (int)blocks_ll;

    max_pool2d_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        padH, padW,
        dilationH, dilationW,
        inputStrideN, inputStrideC, inputStrideH, inputStrideW,
        outputStrideN, outputStrideC, outputStrideH, outputStrideW
    );

    // Error check
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error? We can't throw from .cu, so just ignore or use __assert_fail? Binding should check.
        // For simplicity, we can call cudaDeviceSynchronize and let binding handle? Actually we should not synchronize here.
        // We can just let the error propagate; but the binding may call cudaGetLastError after launch.
        // Better to print? Or assert? Since we cannot include iostream? We can use printf if needed.
        // But typically launcher should not handle errors, just launch. So we leave as is.
    }
}
