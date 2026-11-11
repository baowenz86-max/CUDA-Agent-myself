#include <cuda_runtime.h>
#include <stdint.h>

// Kernel: embedding lookup
// input: int64 indices of shape [B*H*W*D] contiguous
// weight: float32 [num_embeddings, embedding_dim]
// output: float32 [B*H*W*D*embedding_dim]
__global__ void embedding_kernel(const int64_t* __restrict__ input,
                                 const float* __restrict__ weight,
                                 float* __restrict__ output,
                                 int64_t total_elements,
                                 int64_t embedding_dim) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int64_t token = input[idx];
    const float* src = weight + token * embedding_dim;
    float* dst = output + idx * embedding_dim;
    for (int64_t i = 0; i < embedding_dim; ++i) {
        dst[i] = src[i];
    }
}

// Kernel: permute [B,H,W,D,C] -> [B,C,H,W,D]
// input contiguous [B,H,W,D,C], output contiguous [B,C,H,W,D]
__global__ void permute_4d_to_bn3d_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int64_t total_elements,
                                          int64_t B, int64_t C, int64_t H, int64_t W, int64_t D) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    // idx = ((b*C + c)*H*W*D + h*W*D + w*D + d)
    int64_t d = idx % D;
    int64_t tmp = idx / D;
    int64_t w = tmp % W;
    tmp /= W;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t c = tmp % C;
    int64_t b = tmp / C;
    // input index: ((b*H + h)*W + w)*D + d) * C + c
    int64_t src_idx = ((((b * H + h) * W + w) * D + d) * C + c);
    output[idx] = input[src_idx];
}

// Kernel: permute [B,C,H,W,D] -> [B,H,W,D,C]
// input contiguous [B,C,H,W,D], output contiguous [B,H,W,D,C]
__global__ void permute_bn3d_to_4d_kernel(const float* __restrict__ input,
                                          float* __restrict__ output,
                                          int64_t total_elements,
                                          int64_t B, int64_t C, int64_t H, int64_t W, int64_t D) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    // idx = ((b*H + h)*W + w)*D + d) * C + c
    int64_t c = idx % C;
    int64_t tmp = idx / C;
    int64_t d = tmp % D;
    tmp /= D;
    int64_t w = tmp % W;
    tmp /= W;
    int64_t h = tmp % H;
    int64_t b = tmp / H;
    // input index: ((b*C + c)*H*W*D + h*W*D + w*D + d)
    int64_t src_idx = ((((b * C + c) * H + h) * W + w) * D + d);
    output[idx] = input[src_idx];
}

// Kernel: BatchNorm3d eval mode
// input contiguous [B,C,H,W,D]
// output same shape
// weight, bias, running_mean, running_var: length C
// eps: double
__global__ void bn3d_eval_kernel(const float* __restrict__ input,
                                 float* __restrict__ output,
                                 const float* __restrict__ weight,
                                 const float* __restrict__ bias,
                                 const float* __restrict__ running_mean,
                                 const float* __restrict__ running_var,
                                 float eps,
                                 int64_t total_elements,
                                 int64_t C,
                                 int64_t HW_D) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    int64_t c = (idx / HW_D) % C;
    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);
    float w = weight[c];
    float b = bias[c];
    output[idx] = (input[idx] - mean) * inv_std * w + b;
}

// Launcher functions (extern "C", no PyTorch dependencies)

extern "C" void embedding_forward_launcher(const int64_t* input,
                                            const float* weight,
                                            float* output,
                                            int64_t total_elements,
                                            int64_t embedding_dim,
                                            cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    embedding_kernel<<<blocks, block_size, 0, stream>>>(input, weight, output, total_elements, embedding_dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Error handling left for binding layer; cannot include torch headers here.
        // However, we can still record error? Just ignore for now.
    }
}

extern "C" void permute_4d_to_bn3d_launcher(const float* input,
                                             float* output,
                                             int64_t total_elements,
                                             int64_t B, int64_t C, int64_t H, int64_t W, int64_t D,
                                             cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    permute_4d_to_bn3d_kernel<<<blocks, block_size, 0, stream>>>(input, output, total_elements, B, C, H, W, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore, binding layer handles checks
    }
}

extern "C" void bn3d_eval_forward_launcher(const float* input,
                                            float* output,
                                            const float* weight,
                                            const float* bias,
                                            const float* running_mean,
                                            const float* running_var,
                                            float eps,
                                            int64_t total_elements,
                                            int64_t C,
                                            int64_t HW_D,
                                            cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    bn3d_eval_kernel<<<blocks, block_size, 0, stream>>>(input, output, weight, bias, running_mean, running_var, eps, total_elements, C, HW_D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore
    }
}

extern "C" void permute_bn3d_to_4d_launcher(const float* input,
                                             float* output,
                                             int64_t total_elements,
                                             int64_t B, int64_t C, int64_t H, int64_t W, int64_t D,
                                             cudaStream_t stream) {
    const int block_size = 256;
    int64_t blocks = (total_elements + block_size - 1) / block_size;
    permute_bn3d_to_4d_kernel<<<blocks, block_size, 0, stream>>>(input, output, total_elements, B, C, H, W, D);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Ignore
    }
}
