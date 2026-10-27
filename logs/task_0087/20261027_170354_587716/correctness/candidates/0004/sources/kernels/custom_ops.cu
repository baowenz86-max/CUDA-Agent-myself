#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>

// CUDA kernels and launchers for model_new.py custom operations.
// This file must not include torch/ATen/pybind headers.
// All launchers use raw pointers and cudaStream_t.

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        return; \
    } \
} while(0)

// -----------------------------------------------------------------------------
// group_norm_forward
// input x: [N, C, L] where N=512, C=64, L=20, contiguous float32
// weight, bias: [C]
// num_groups = 4, num_channels = 64, eps = 1e-5
// output: same shape as x, normalized per group over channels and spatial
// -----------------------------------------------------------------------------

__global__ void group_norm_kernel(const float* __restrict__ x,
                                  const float* __restrict__ weight,
                                  const float* __restrict__ bias,
                                  float* __restrict__ y,
                                  int N, int C, int L,
                                  int num_groups, float eps) {
    int group_size = C / num_groups; // 16
    int spatial = L; // 20
    int total_per_group = group_size * spatial; // 320

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= N * C * L) return;

    int n = index / (C * L);
    int rem = index % (C * L);
    int c = rem / L;
    int l = rem % L;
    int g = c / group_size;

    // Compute mean and variance for this (n, g) over group channels and spatial
    // Use one thread per element: not efficient but simple; can optimize later.
    // For each element, we compute mean/var by looping (redundant but correct).
    float mean = 0.0f;
    float var = 0.0f;
    int group_start_c = g * group_size;
    for (int cc = group_start_c; cc < group_start_c + group_size; ++cc) {
        for (int ll = 0; ll < L; ++ll) {
            float val = x[n * C * L + cc * L + ll];
            mean += val;
            var += val * val;
        }
    }
    mean /= total_per_group;
    var = var / total_per_group - mean * mean;
    float inv_std = rsqrtf(var + eps);

    y[index] = (x[index] - mean) * inv_std * weight[c] + bias[c];
}

// launcher for group_norm
// All pointers are device pointers, stream is cudaStream_t
extern "C" void group_norm_forward_cuda(
    const float* x, const float* weight, const float* bias,
    float* y, int N, int C, int L, int num_groups, float eps,
    cudaStream_t stream) {
    int total = N * C * L;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    group_norm_kernel<<<blocks, threads, 0, stream>>>(x, weight, bias, y, N, C, L, num_groups, eps);
}

// -----------------------------------------------------------------------------
// scalar_div
// input x: [N, C, L] contiguous float32 (or any shape, contiguous after permute?)
// output y = x / divisor elementwise, same shape
// -----------------------------------------------------------------------------

__global__ void scalar_div_kernel(const float* __restrict__ x,
                                  float* __restrict__ y,
                                  int numel, float divisor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        y[idx] = x[idx] / divisor;
    }
}

extern "C" void scalar_div_cuda(
    const float* x, float* y, int numel, float divisor, cudaStream_t stream) {
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    scalar_div_kernel<<<blocks, threads, 0, stream>>>(x, y, numel, divisor);
}

// -----------------------------------------------------------------------------
// LSTM forward using cuBLAS
// We implement a multi-layer LSTM manually.
// Input x: [batch, seq_len, input_size] contiguous -> batch=512, seq=20, input=64
// h0,c0: [num_layers, batch, hidden_size] -> [2,512,128]
// weights: weight_ih_l0 [4*hidden, input], weight_hh_l0 [4*hidden, hidden], bias_ih_l0 [4*hidden], bias_hh_l0 [4*hidden]
// similarly for layer1 with input_size=hidden_size=128
// Output: [batch, seq_len, hidden_size] -> [512,20,128]
// h_n, c_n: [num_layers, batch, hidden_size]
// Note: PyTorch's LSTM uses gates order i, f, g, o (where g is tanh candidate).
// We'll use cublasSgemm for matrix multiplies.
// We need temporary buffers for gate pre-activations and hidden states.
// Since launcher cannot return tensors, binding layer will allocate output, h_n, c_n and pass pointers.
// -----------------------------------------------------------------------------

// Elementwise kernel to compute gates from pre-activations and update cell state and hidden state.
// pre_gates: [batch, 4*hidden] for a given time step
// c_prev, h_prev: (never used? we compute c_t and h_t)
// c_t, h_t: outputs for current step [batch, hidden]
// hidden_size H
__global__ void lstm_cell_kernel(const float* pre_gates, // [batch, 4H]
                                 const float* c_prev,    // [batch, H]
                                 float* c_t,             // [batch, H]
                                 float* h_t,             // [batch, H]
                                 int batch, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * H) return;
    int b = idx / H;
    int h = idx % H;

    float i_gate = 1.0f / (1.0f + expf(-pre_gates[b * 4 * H + h]));
    float f_gate = 1.0f / (1.0f + expf(-pre_gates[b * 4 * H + H + h]));
    float g_gate = tanhf(pre_gates[b * 4 * H + 2 * H + h]);
    float o_gate = 1.0f / (1.0f + expf(-pre_gates[b * 4 * H + 3 * H + h]));

    float c_val = f_gate * c_prev[idx] + i_gate * g_gate;
    c_t[idx] = c_val;
    h_t[idx] = o_gate * tanhf(c_val);
}

// Kernel to add bias to a matrix row-wise: y = x + bias (broadcast over rows)
// y: [rows, cols], bias: [cols]
__global__ void add_bias_kernel(const float* x, const float* bias, float* y, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int col = idx % cols;
    y[idx] = x[idx] + bias[col];
}

// LSTM launcher using cuBLAS
// Parameters:
// x: [batch, seq_len, input_size]
// weights: for each layer, weight_ih [4H, input], weight_hh [4H, H], bias_ih [4H], bias_hh [4H]
// h0,c0: [num_layers, batch, H]
// output: [batch, seq_len, H]
// h_n, c_n: [num_layers, batch, H]
// All pointers device, stream for cublas and kernels
extern "C" void lstm_forward_cuda(
    const float* x, // [batch, seq, input_size]
    const float* w_ih_l0, const float* w_hh_l0, const float* b_ih_l0, const float* b_hh_l0,
    const float* w_ih_l1, const float* w_hh_l1, const float* b_ih_l1, const float* b_hh_l1,
    const float* h0, const float* c0,
    float* output, float* h_n, float* c_n,
    int batch, int seq_len, int input_size, int hidden_size, int num_layers,
    cudaStream_t stream) {

    // For simplicity, assume num_layers=2, batch_first true, input_size=64, hidden_size=128, seq=20
    // We'll implement two-layer loop.
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);

    const float alpha = 1.0f, beta = 0.0f;

    int H = hidden_size;
    int D = input_size;
    int B = batch;
    int S = seq_len;

    // Allocate temporary buffers for gates and hidden states for each layer/time.
    // We'll process sequentially, but need to store all output hidden states for output.
    // We also need intermediate hidden states for next layer input.

    // Allocate device memory for first layer hidden states and cell states for all time steps? 
    // For memory efficiency, we can process layer by layer and store hidden states as needed.
    // For first layer: input x is [B,S,D], output h1_all [B,S,H], final h1_n [B,H], c1_n [B,H]
    // For second layer: input h1_all [B,S,H], output h2_all [B,S,H] which is final output, final states.
    // We'll allocate temporary arrays:
    float *pre_gates, *c_prev, *c_cur, *h_cur;
    cudaMalloc(&pre_gates, B * 4 * H * sizeof(float));
    cudaMalloc(&c_prev, B * H * sizeof(float));
    cudaMalloc(&c_cur, B * H * sizeof(float));
    cudaMalloc(&h_cur, B * H * sizeof(float));

    // For first layer, intermediate hidden states h1_all [B,S,H] might be large: 512*20*128 = 1,310,720 floats ~5MB, okay.
    float *h1_all;
    cudaMalloc(&h1_all, B * S * H * sizeof(float));

    // Layer 0
    {
        // initial states h0[0], c0[0]
        // copy c_prev from c0 layer 0
        cudaMemcpyAsync(c_prev, c0 + 0 * B * H, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        // initial h_prev not needed for computation except for W_hh, which uses h_prev.
        // We need h_prev for W_hh multiplication; use h_cur from previous step or h0[0] for first step.
        cudaMemcpyAsync(h_cur, h0 + 0 * B * H, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        for (int t = 0; t < S; ++t) {
            // Input at time t: x_t is x[:, t, :], dimensions B x D
            const float* x_t = x + t * D; // but x is B x S x D contiguous, so column stride? Actually x is [B,S,D] contiguous, so x[b,t,d] = x[b*S*D + t*D + d]. So pointer to all batch for time t is not contiguous; we need to gather. 
            // Better to use cublas with leading dimensions.
            // We can do gemm: pre_gates = x_t * W_ih^T + b_ih + h_prev * W_hh^T + b_hh
            // x_t matrix is B x D, but x_t is not contiguous because stride between batches is S*D.
            // However, we can use cublasSgemm with lda = S*D, ldb = D, ldc = 4H.
            // Actually cublasSgemm computes C = alpha * A * B + beta * C, where A is m x k, B is k x n, C is m x n.
            // We want pre_gates = x_t (B x D) * W_ih^T (D x 4H) => C = B x 4H.
            // We can set m=B, n=4H, k=D, A = x + t*D (but with lda = S*D), B = w_ih_l0 (row-major, 4H x D, but we need D x 4H, so we can use transpose).
            // For simplicity, use cublasSgemm with transa=CUBLAS_OP_N, transb=CUBLAS_OP_T.
            // B = w_ih_l0 is 4H x D, so B^T is D x 4H, exactly what we need.
            // lda for A: if A is B x D but stored in larger x, lda = S*D.
            // ldb for B: w_ih_l0 is 4H x D, so ldb = D (since row-major, row stride = D).
            // ldc = 4H.
            // We'll accumulate into pre_gates, starting with 0.
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        B, 4*H, D,
                        &alpha,
                        x + t*D, S*D,
                        w_ih_l0, D,
                        &beta,
                        pre_gates, 4*H);
            // Add bias_ih_l0 to pre_gates (row-wise)
            int total_gates = B * 4 * H;
            int threads_blk = 256;
            int blocks_bias = (total_gates + threads_blk - 1) / threads_blk;
            add_bias_kernel<<<blocks_bias, threads_blk, 0, stream>>>(pre_gates, b_ih_l0, pre_gates, B, 4*H);

            // Now h_prev * W_hh^T: h_prev is B x H, W_hh is 4H x H, so result B x 4H.
            // Use cublasSgemm with transa=N, transb=T, k=H.
            float one=1.0f;
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        B, 4*H, H,
                        &alpha,
                        h_cur, H,   // lda for h_cur: H because contiguous B x H, row stride H
                        w_hh_l0, H,
                        &one,  // beta=1 to accumulate
                        pre_gates, 4*H);
            add_bias_kernel<<<blocks_bias, threads_blk, 0, stream>>>(pre_gates, b_hh_l0, pre_gates, B, 4*H);

            // Compute cell and hidden states
            lstm_cell_kernel<<<blocks_bias, threads_blk, 0, stream>>>(pre_gates, c_prev, c_cur, h_cur, B, H);

            // Copy h_cur to h1_all[:, t, :]
            // h1_all is B x S x H, to store h_cur at time t: h1_all + t*H with stride S*H? Actually B x S x H contiguous, so h1_all[b,t,h] = h1_all[b*S*H + t*H + h]. So we need to copy h_cur (B x H) into columns t of each batch.
            // We can write a kernel to copy strided.
            // Or use cudaMemcpy2DAsync for each batch? Simpler: use a custom kernel to copy.
            // We'll define copy_hidden_to_output_kernel later, but for brevity here we can use cudaMemcpy2DAsync.
            // Since h_cur is contiguous B x H, and destination for time t is h1_all + t*H, with row stride S*H, col stride 1.
            // We can copy each row of batch separately using cudaMemcpyAsync.
            for (int b = 0; b < B; ++b) {
                cudaMemcpyAsync(h1_all + b * S * H + t * H, h_cur + b * H, H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }

            // Update c_prev = c_cur for next time step
            cudaMemcpyAsync(c_prev, c_cur, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
        // After loop, h_cur and c_cur are final states for layer0
        // Copy to h_n[0], c_n[0]
        cudaMemcpyAsync(h_n + 0 * B * H, h_cur, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(c_n + 0 * B * H, c_cur, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // Layer 1: input is h1_all [B,S,H], output h2_all [B,S,H] which becomes final output tensor.
    // We can reuse output pointer for h2_all.
    {
        // Reset c_prev, h_cur from layer1 initial states
        cudaMemcpyAsync(c_prev, c0 + 1 * B * H, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(h_cur, h0 + 1 * B * H, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        for (int t = 0; t < S; ++t) {
            // x_t for layer1 is h1_all[:, t, :], B x H, with stride S*H between batch rows.
            const float* x_t = h1_all + t * H; // but again not contiguous across batch
            // gemm: pre_gates = x_t * W_ih1^T + b_ih1 + h_prev * W_hh1^T + b_hh1
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        B, 4*H, H,
                        &alpha,
                        x_t, S*H,
                        w_ih_l1, H,
                        &beta,
                        pre_gates, 4*H);
            int total_gates = B * 4 * H;
            int threads_blk = 256;
            int blocks_bias = (total_gates + threads_blk - 1) / threads_blk;
            add_bias_kernel<<<blocks_bias, threads_blk, 0, stream>>>(pre_gates, b_ih_l1, pre_gates, B, 4*H);

            float one=1.0f;
            cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                        B, 4*H, H,
                        &alpha,
                        h_cur, H,
                        w_hh_l1, H,
                        &one,
                        pre_gates, 4*H);
            add_bias_kernel<<<blocks_bias, threads_blk, 0, stream>>>(pre_gates, b_hh_l1, pre_gates, B, 4*H);

            lstm_cell_kernel<<<blocks_bias, threads_blk, 0, stream>>>(pre_gates, c_prev, c_cur, h_cur, B, H);

            // Copy h_cur to output[:, t, :] (output is B x S x H)
            for (int b = 0; b < B; ++b) {
                cudaMemcpyAsync(output + b * S * H + t * H, h_cur + b * H, H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
            }
            cudaMemcpyAsync(c_prev, c_cur, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
        cudaMemcpyAsync(h_n + 1 * B * H, h_cur, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(c_n + 1 * B * H, c_cur, B * H * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    cudaFree(pre_gates);
    cudaFree(c_prev);
    cudaFree(c_cur);
    cudaFree(h_cur);
    cudaFree(h1_all);
    cublasDestroy(cublas_handle);
}
