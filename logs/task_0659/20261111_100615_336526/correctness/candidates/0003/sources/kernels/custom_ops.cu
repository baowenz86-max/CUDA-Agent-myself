#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Forward declarations of CUDA kernels used by launchers
__global__ void copy_strided_kernel(float* dst, const float* src, int rows, int cols, int src_stride_0, int src_stride_1);
__global__ void add_bias_kernel(float* output, const float* bias, int rows, int cols);
__global__ void attention_kernel(const float* __restrict__ query, const float* __restrict__ key, const float* __restrict__ value, float* __restrict__ output, int N, int S, int E, float scale, int out_stride_0, int out_stride_1, int out_stride_2);
__global__ void concat_kernel(const float* __restrict__ x1, const float* __restrict__ x2, float* __restrict__ output, int N, int S, int E, int x1_stride_0, int x1_stride_1, int x2_stride_0, int x2_stride_1);

// 确保 launcher 使用 C 链接，供绑定层调用
extern "C" {

// -----------------------------------------------------------------------------
// linear_forward: output = input @ weight^T + bias (可选)
// input: [N, S, E], weight: [E, E], bias: [E] 或 nullptr
// 输出: [N, S, E]
// 使用 cuBLAS Sgemm，输入非连续时先拷贝到临时连续缓冲
void linear_forward_launcher(
    const float* input,      // [N*S, E] 逻辑视图，步长 strided
    const float* weight,     // [E, E] 连续
    const float* bias,       // [E] 或 nullptr
    float* output,           // [N*S, E] 连续
    int N, int S, int E,
    int input_stride_0,      // 行步长（元素数），通常是 S*E 或 E
    int input_stride_1,      // 列步长，通常为 1
    cudaStream_t stream)
{
    // 检查是否需要拷贝输入到连续缓冲区（如果 stride_1 != 1 或 stride_0 != E）
    const bool need_copy = (input_stride_1 != 1 || input_stride_0 != E);
    float* contiguous_input = nullptr;
    if (need_copy) {
        cudaMalloc(&contiguous_input, (size_t)N * S * E * sizeof(float));
        // 拷贝 kernel，处理任意步长
        dim3 block(256);
        dim3 grid((N * S + 255) / 256);
        // 使用 lambda 或内联 kernel，这里写一个拷贝 kernel
        // 为避免复杂，在 host 端使用 cudaMemcpy2D 不合适，这里用自定义 kernel
        // 定义在下方
        copy_strided_kernel<<<grid, block, 0, stream>>>(
            contiguous_input, input, N*S, E, input_stride_0, input_stride_1);
    } else {
        contiguous_input = const_cast<float*>(input); // 如果连续，直接使用
    }

    // cuBLAS 句柄
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    // 计算 output = input @ weight^T
    // 注意：PyTorch Linear 是 input @ weight.T
    // cuBLAS column-major 表示：C = alpha * A * B + beta * C
    // 这里我们想要 C(row-major, M x N) = A(row-major, M x K) * B(row-major, K x N)
    // 其中 B = weight^T，即 weight 转置
    // 等效于 column-major: C^T = B^T * A^T
    // 使用 cublasSgemm 的 transa/transb 参数。
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int m = N * S;       // 行数
    int n = E;           // 列数
    int k = E;           // 内积维度

    // 由于 weight 在内存中是 row-major [E,E]，但 cuBLAS 默认 column-major，
    // 我们需要将 weight 解释为 column-major 的 [E,E] 转置矩阵，即 transb = CUBLAS_OP_T 或使用 weight.T。
    // 常见做法：cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &alpha,
    //                input, m, weight, n, &beta, output, m);
    // 其中 input 是 column-major 的 m x k，weight 是 column-major 的 n x k，取转置后得到 k x n。
    // 由于内存 row-major，这等价于计算 input @ weight^T。
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, k,
                &alpha,
                contiguous_input, m,
                weight, n,
                &beta,
                output, m);

    cublasDestroy(handle);

    // 加 bias
    if (bias != nullptr) {
        dim3 bias_block(256);
        dim3 bias_grid((m + 255) / 256);
        add_bias_kernel<<<bias_grid, bias_block, 0, stream>>>(
            output, bias, m, E);
    }

    if (need_copy) {
        cudaFree(contiguous_input);
    }
}

// -----------------------------------------------------------------------------
// scaled_dot_product_attention_forward
// query: [N, S, E], key: [N, S, E], value: [N, S, E]
// 输出 attn: [N, S, E]，但 stride 可能非连续（参考为 [512, 8192, 1]）
// 我们需要输出张量由绑定层分配，其 stride 由绑定层设定。
// launcher 接收输出指针和其 stride 信息，不负责分配。
// 参数：
//   query, key, value: 输入指针 (连续)
//   output: 输出指针，stride 由 output_stride_0, output_stride_1, output_stride_2 给出
//   N, S, E
//   实际参考输出 stride: [512, 8192, 1] 奇怪，但我们必须遵循。
//   我们将根据给定 stride 写入。
void scaled_dot_product_attention_forward_launcher(
    const float* query,   // [N, S, E]
    const float* key,     // [N, S, E]
    const float* value,   // [N, S, E]
    float* output,        // [N, S, E]
    int N, int S, int E,
    int output_stride_0,  // 通常 S*E 或 512 (参考)
    int output_stride_1,  // 通常 E 或 8192 (参考)
    int output_stride_2,  // 通常 1
    cudaStream_t stream)
{
    // 实现 scaled dot product attention，无 mask
    // 每个 block 处理一个 (n,s) 位置，计算该行与所有 key 的相似度，softmax，加权 value
    dim3 block(256);
    dim3 grid(N * S);
    const float scale = 1.0f / sqrtf((float)E);

    // 每个线程块处理一对 (n, s)
    attention_kernel<<<grid, block, 0, stream>>>(
        query, key, value, output,
        N, S, E, scale,
        output_stride_0, output_stride_1, output_stride_2);
}

// -----------------------------------------------------------------------------
// concat_forward: output = concat(x1, x2, dim=-1)
// x1: [N, S, E], x2: [N, S, E], output: [N, S, 2E]
// 输出连续，由绑定层分配。
void concat_forward_launcher(
    const float* x1,       // [N, S, E]
    const float* x2,       // [N, S, E]
    float* output,         // [N, S, 2E]
    int N, int S, int E,
    int x1_stride_0, int x1_stride_1,
    int x2_stride_0, int x2_stride_1,
    cudaStream_t stream)
{
    int total_elements = N * S * (2 * E);
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    concat_kernel<<<blocks, threads, 0, stream>>>(
        x1, x2, output,
        N, S, E,
        x1_stride_0, x1_stride_1,
        x2_stride_0, x2_stride_1);
}

} // extern "C"

// -----------------------------------------------------------------------------
// Kernel 定义
// -----------------------------------------------------------------------------

// 拷贝 strided 输入到连续缓冲区
__global__ void copy_strided_kernel(
    float* dst, const float* src,
    int rows, int cols,
    int src_stride_0, int src_stride_1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    dst[idx] = src[r * src_stride_0 + c * src_stride_1];
}

// 加 bias
__global__ void add_bias_kernel(
    float* output, const float* bias,
    int rows, int cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    output[idx] += bias[col];
}

// 注意力 kernel：每个 block 处理一个 (n,s)
__global__ void attention_kernel(
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    float* __restrict__ output,
    int N, int S, int E,
    float scale,
    int out_stride_0, int out_stride_1, int out_stride_2)
{
    int n = blockIdx.x / S;
    int s = blockIdx.x % S;

    // 共享内存用于存储该行的 scores 和 softmax 结果
    extern __shared__ float scores[]; // size S
    int tid = threadIdx.x;

    // 计算 scores
    const float* q_row = query + (n * S + s) * E;
    const float* k_mat = key + n * S * E;

    float local_score = 0.0f;
    for (int i = tid; i < S; i += blockDim.x) {
        const float* k_row = k_mat + i * E;
        float dot = 0.0f;
        for (int j = 0; j < E; ++j) {
            dot += q_row[j] * k_row[j];
        }
        scores[i] = dot * scale;
    }
    __syncthreads();

    // 计算 softmax（数值稳定）
    // 求最大值
    float max_val = -1e30f;
    for (int i = tid; i < S; i += blockDim.x) {
        if (scores[i] > max_val) max_val = scores[i];
    }
    // 块内归约求最大
    __shared__ float block_max;
    if (tid == 0) block_max = max_val;
    __syncthreads();
    // 简化：使用原子操作或者直接让一个线程做归约，S 最大 1024，block 256，可以用 warp shuffle 优化
    // 这里使用简单的跨线程共享内存归约
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (block_max < max_val) block_max = max_val;
        }
        __syncthreads();
    }
    max_val = block_max;

    // 计算 exp 和 sum
    float sum = 0.0f;
    for (int i = tid; i < S; i += blockDim.x) {
        scores[i] = expf(scores[i] - max_val);
        sum += scores[i];
    }
    // 归约 sum
    __shared__ float block_sum;
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();
    atomicAdd(&block_sum, sum);
    __syncthreads();
    sum = block_sum;

    // 计算输出：out = sum_i scores[i] * value[i]
    const float* v_mat = value + n * S * E;
    float* out_row = output + n * out_stride_0 + s * out_stride_1;

    for (int j = tid; j < E; j += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < S; ++i) {
            acc += scores[i] * v_mat[i * E + j];
        }
        out_row[j * out_stride_2] = acc / sum;
    }
}

// 拼接 kernel
__global__ void concat_kernel(
    const float* __restrict__ x1,
    const float* __restrict__ x2,
    float* __restrict__ output,
    int N, int S, int E,
    int x1_stride_0, int x1_stride_1,
    int x2_stride_0, int x2_stride_1)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * S * (2 * E);
    if (idx >= total) return;

    int n = idx / (S * 2 * E);
    int rem = idx % (S * 2 * E);
    int s = rem / (2 * E);
    int c = rem % (2 * E);

    const float* src;
    int src_index;
    if (c < E) {
        // 从 x1 复制
        src = x1;
        src_index = (n * S + s) * x1_stride_0 + c * x1_stride_1;
    } else {
        // 从 x2 复制
        src = x2;
        src_index = (n * S + s) * x2_stride_0 + (c - E) * x2_stride_1;
    }
    output[idx] = src[src_index];
}
