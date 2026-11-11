#include <cuda_runtime.h>
#include <stdint.h>

// embedding_forward: output[b,h,e] = weight[input[b,h], e]
// input: int64 [B,H], weight: float [V,E], output: float [B,H,E]
__global__ void embedding_lookup_kernel(
    const int64_t* __restrict__ input,
    const float* __restrict__ weight,
    float* __restrict__ output,
    int B, int H, int E,
    long long total_elements) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (; idx < total_elements; idx += stride) {
        int e = (int)(idx % E);
        long long tmp = idx / E;
        int h = (int)(tmp % H);
        int b = (int)(tmp / H);
        // input is contiguous [B,H]
        int64_t token = input[(long long)b * H + h];
        output[idx] = weight[token * E + e];
    }
}

// avg_pool1d_forward: average pooling over last dim (length L)
// input shape [B,C,L], output shape [B,C,outL]
// kernel_size K, stride S, padding P (P=0 in this task)
// input may be non-contiguous with arbitrary strides: in_stride_b, in_stride_c, in_stride_l
// output is contiguous
__global__ void avg_pool1d_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int B, int C, int L,
    int K, int S, int P,
    int outL,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_l) {
    long long idx = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long total = (long long)B * C * outL;
    long long grid_stride = (long long)gridDim.x * blockDim.x;
    float inv_k = 1.0f / (float)K;
    for (; idx < total; idx += grid_stride) {
        int ol = (int)(idx % outL);
        long long tmp = idx / outL;
        int c = (int)(tmp % C);
        int b = (int)(tmp / C);

        // input start index along length: ol * S - P
        int l_start = ol * S - P;
        float sum = 0.0f;
        // exactly K elements are always included when padding=0 and valid output positions
        // For general P, some positions may include padding zeros; we simply skip negative indices
        int valid_count = 0;
        for (int k = 0; k < K; ++k) {
            int l = l_start + k;
            if (l >= 0 && l < L) {
                const float* in_ptr = input + (long long)b * in_stride_b + (long long)c * in_stride_c + (long long)l * in_stride_l;
                sum += *in_ptr;
                valid_count++;
            }
        }
        // In PyTorch AvgPool1d, count_include_pad=True by default, so divisor is K even if padding zeros included
        // However, if valid_count < K due to padding, we still divide by K to match semantics.
        // For safety, if K == 0 (invalid), avoid division by zero.
        float avg = (K > 0) ? (sum * inv_k) : 0.0f;
        output[idx] = avg;
    }
}

// Launcher for embedding_forward
extern "C" void embedding_forward_launcher(
    const int64_t* input,
    const float* weight,
    float* output,
    int B, int H, int E,
    cudaStream_t stream) {
    long long total = (long long)B * H * E;
    int block = 256;
    // grid dim limited to avoid overflow; use enough blocks to cover but not exceed 2^31-1
    long long max_blocks = 2147483647LL; // 2^31 - 1
    long long needed = (total + block - 1) / block;
    int grid = (int)(needed < max_blocks ? needed : max_blocks);
    embedding_lookup_kernel<<<grid, block, 0, stream>>>(input, weight, output, B, H, E, total);
}

// Launcher for avg_pool1d_forward
extern "C" void avg_pool1d_forward_launcher(
    const float* input,
    float* output,
    int B, int C, int L,
    int kernel_size, int stride, int padding,
    int outL,
    long long in_stride_b,
    long long in_stride_c,
    long long in_stride_l,
    cudaStream_t stream) {
    long long total = (long long)B * C * outL;
    int block = 256;
    long long max_blocks = 2147483647LL;
    long long needed = (total + block - 1) / block;
    int grid = (int)(needed < max_blocks ? needed : max_blocks);
    avg_pool1d_kernel<<<grid, block, 0, stream>>>(
        input, output, B, C, L, kernel_size, stride, padding, outL,
        in_stride_b, in_stride_c, in_stride_l);
}
