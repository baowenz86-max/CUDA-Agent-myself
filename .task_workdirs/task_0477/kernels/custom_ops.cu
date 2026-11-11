#include <cuda_runtime.h>

// InstanceNorm3d normalization kernel: y = (x - mean[c]) / sqrt(var[c] + eps)
// x is assumed to be at most 3D? Actually 5D N,C,D,H,W, but we treat as flattened with strides.
// To support arbitrary strides, we pass stride for each dimension? Simpler: pass a pointer to x and total numel,
// but we need channel index to fetch mean/var. Since x may be non-contiguous, we need to compute channel from linear
// index based on strides. We'll pass strides for N and C (and assume D,H,W are contiguous or also pass all strides).
// Contract says x layout supports strides, so we pass all strides: strideN, strideC, strideD, strideH, strideW.
// For simplicity, we treat x as contiguous in memory? But contract explicit says supports strides.
// However reference input is contiguous, but to be safe we will handle arbitrary strides by computing flat index from
// multidimensional indices using provided strides.
// The launcher will take pointer to x, output, and all strides. We'll compute total number of elements as N*C*D*H*W.

template <typename T>
__global__ void instance_norm3d_kernel(
    T* __restrict__ output,
    const T* __restrict__ x,
    const T* __restrict__ running_mean,
    const T* __restrict__ running_var,
    T eps,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t strideN,
    int64_t strideC,
    int64_t strideD,
    int64_t strideH,
    int64_t strideW
) {
    int64_t total = N * C * D * H * W;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    for (int64_t i = idx; i < total; i += stride) {
        // Decompose linear index into n,c,d,h,w using strides? Need to know dimensions ordering.
        // We assume standard N,C,D,H,W order, so we can compute:
        int64_t tmp = i;
        int64_t w = tmp % W;
        tmp /= W;
        int64_t h = tmp % H;
        tmp /= H;
        int64_t d = tmp % D;
        tmp /= D;
        int64_t c = tmp % C;
        int64_t n = tmp / C;
        
        int64_t input_offset = n * strideN + c * strideC + d * strideD + h * strideH + w * strideW;
        T mean = running_mean[c];
        T var = running_var[c];
        T inv_std = rsqrtf(var + eps);
        output[i] = (x[input_offset] - mean) * inv_std;
    }
}

extern "C" void instance_norm3d_forward_launcher(
    float* output,
    const float* x,
    const float* running_mean,
    const float* running_var,
    double eps,
    int64_t N,
    int64_t C,
    int64_t D,
    int64_t H,
    int64_t W,
    int64_t strideN,
    int64_t strideC,
    int64_t strideD,
    int64_t strideH,
    int64_t strideW,
    cudaStream_t stream
) {
    int64_t total = N * C * D * H * W;
    if (total <= 0) return;
    const int threads = 256;
    int64_t blocks = (total + threads - 1) / threads;
    // Cap blocks to 65535 for older GPUs? We'll use grid-stride with max grid dimension.
    // For simplicity, use a reasonable number of blocks (e.g., min(blocks, 65535)).
    // Since total is huge (169M), blocks would be ~662k, which exceeds max grid dim in x. Use grid-stride with a grid size
    // limited to 2^31-1 but practically we set grid = min(blocks, 65535) and grid-stride loop handles rest.
    int64_t grid = blocks > 65535 ? 65535 : blocks;
    instance_norm3d_kernel<float><<<grid, threads, 0, stream>>>(
        output, x, running_mean, running_var, (float)eps,
        N, C, D, H, W, strideN, strideC, strideD, strideH, strideW
    );
}
