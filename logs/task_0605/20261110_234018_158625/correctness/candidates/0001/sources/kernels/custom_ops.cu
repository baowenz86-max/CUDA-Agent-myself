#include <cuda_runtime.h>
#include <math.h>

// MaxPool2d forward kernel for NCHW float32 tensor
// Supports arbitrary stride, padding, dilation, ceil_mode (ceil_mode only affects output size, but we follow ceil_mode semantics)
// Ignores padding regions by initializing to -inf and only considering valid input coordinates.
__global__ void maxpool2d_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int paddingH, int paddingW,
    int dilationH, int dilationW) {
    
    int total = N * C * outH * outW;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total; i += stride) {
        int ow = i % outW;
        int tmp = i / outW;
        int oh = tmp % outH;
        tmp = tmp / outH;
        int c = tmp % C;
        int n = tmp / C;
        
        // Input start coordinates corresponding to output (oh, ow) before padding
        int hstart = oh * strideH - paddingH;
        int wstart = ow * strideW - paddingW;
        
        float maxval = -INFINITY;
        
        for (int kh = 0; kh < kernelH; ++kh) {
            int ih = hstart + kh * dilationH;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < kernelW; ++kw) {
                int iw = wstart + kw * dilationW;
                if (iw < 0 || iw >= W) continue;
                const float val = input[((n * C + c) * H + ih) * W + iw];
                if (val > maxval) maxval = val;
            }
        }
        
        // If no valid element (should not happen with valid output indices), output -inf, but PyTorch would produce -inf for all-padding windows? Actually such windows are not generated because output size only considers positions where kernel overlaps at least partially.
        // However, with ceil_mode and padding, some output positions may have no valid input elements? MaxPool with count_include_pad=False ignores padding, but if all positions are padding, the output is -inf? PyTorch documentation says output is -inf for empty windows if ceil_mode used? Let's check: In practice, output size formula ensures that at least one valid element exists for any output position when ceil_mode=False (floor). For ceil_mode=True, output may include positions beyond the input, causing empty windows. Since current model uses ceil_mode=False, we can safely assume at least one valid element. But to be safe, we output maxval even if -inf.
        output[i] = maxval;
    }
}

// ReLU forward kernel: output = max(input, 0)
__global__ void relu_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int total) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < total; i += stride) {
        float val = input[i];
        output[i] = val > 0.0f ? val : 0.0f;
    }
}

// Launcher for maxpool2d
// All tensors are assumed contiguous. If input may be non-contiguous, binding layer should pass contiguous copy or handle strides. 
// The contract mentions "layout: supports strides", but launcher receives raw pointer; cannot handle strides. Therefore binding layer must ensure contiguity, or we can pass strides. We'll design launcher to accept only contiguous data pointer, and binding will call .contiguous() if needed.
// The binding.cpp will be responsible for making contiguous.
extern "C" void maxpool2d_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int outH, int outW,
    int kernelH, int kernelW,
    int strideH, int strideW,
    int paddingH, int paddingW,
    int dilationH, int dilationW,
    cudaStream_t stream) {
    
    int total = N * C * outH * outW;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // Limit blocks to a reasonable number; grid-stride loop handles rest
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    
    maxpool2d_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output,
        N, C, H, W,
        outH, outW,
        kernelH, kernelW,
        strideH, strideW,
        paddingH, paddingW,
        dilationH, dilationW);
}

// Launcher for relu
extern "C" void relu_launcher(
    const float* input,
    float* output,
    int total,
    cudaStream_t stream) {
    
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    const int max_blocks = 65535;
    if (blocks > max_blocks) blocks = max_blocks;
    
    relu_forward_kernel<<<blocks, threads, 0, stream>>>(
        input, output, total);
}
