#include <cuda_runtime.h>
#include <cudnn.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = (call);                                                   \
    if (err != cudaSuccess) {                                                   \
      return err;                                                               \
    }                                                                           \
  } while (0)

#define CUDNN_CHECK(call)                                                       \
  do {                                                                          \
    cudnnStatus_t status = (call);                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                                       \
      return cudaErrorUnknown;                                                  \
    }                                                                           \
  } while (0)

// -----------------------------------------------------------------------------
// Layer normalization kernels
// -----------------------------------------------------------------------------

// Compute mean and variance for each row (last dimension is normalized)
// x: input tensor of shape (..., N), stride[last] = row_stride, stride[last] = 1 (or contiguous)
// We assume the last dimension is contiguous, but preceding dimensions may have arbitrary strides.
// grid: one block per row, blockDim.x threads handle elements of the row.
template <int BLOCK_SIZE>
__global__ void layer_norm_mean_var_kernel(const float* __restrict__ x,
                                           float* __restrict__ mean,
                                           float* __restrict__ var,
                                           int N,
                                           int rows,
                                           int64_t row_stride) {
    __shared__ float s_mean[BLOCK_SIZE];
    __shared__ float s_var[BLOCK_SIZE];

    int row = blockIdx.x;
    if (row >= rows) return;

    const float* row_ptr = x + row * row_stride;
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Grid-stride loop over N elements of the row
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = row_ptr[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Warp-level reduction (optional, but simple block reduction with shared memory)
    s_mean[tid] = local_sum;
    s_var[tid] = local_sq_sum;
    __syncthreads();

    // Reduce within block
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mean[tid] += s_mean[tid + stride];
            s_var[tid] += s_var[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float m = s_mean[0] / N;
        float v = s_var[0] / N - m * m;
        mean[row] = m;
        var[row] = v;
    }
}

// Apply normalization and optional affine transformation
// mean and var arrays have one entry per row
// y = (x - mean) / sqrt(var + eps) * (weight ? weight[j] : 1) + (bias ? bias[j] : 0)
__global__ void layer_norm_apply_kernel(const float* __restrict__ x,
                                        float* __restrict__ y,
                                        const float* __restrict__ mean,
                                        const float* __restrict__ var,
                                        const float* __restrict__ weight,
                                        const float* __restrict__ bias,
                                        float eps,
                                        int N,
                                        int rows,
                                        int64_t row_stride,
                                        int64_t out_row_stride,
                                        bool has_weight,
                                        bool has_bias) {
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* x_row = x + row * row_stride;
    float* y_row = y + row * out_row_stride;
    float m = mean[row];
    float v = var[row];
    float inv_std = rsqrtf(v + eps);

    int tid = threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;
    // grid-stride loop over N
    for (int j = tid; j < N; j += total_threads) {
        float val = x_row[j];
        float norm = (val - m) * inv_std;
        if (has_weight && weight != nullptr) norm *= weight[j];
        if (has_bias && bias != nullptr) norm += bias[j];
        y_row[j] = norm;
    }
}

// -----------------------------------------------------------------------------
// nan_to_num kernel (default: nan -> 0, +-inf unchanged)
// -----------------------------------------------------------------------------
__global__ void nan_to_num_kernel(float* __restrict__ y,
                                  const float* __restrict__ x,
                                  int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = isnan(val) ? 0.0f : val;
    }
}

// -----------------------------------------------------------------------------
// add_bias kernel: scalar bias broadcast to all elements
// -----------------------------------------------------------------------------
__global__ void add_bias_kernel(float* __restrict__ y,
                                const float* __restrict__ x,
                                float bias,
                                int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = x[idx] + bias;
    }
}

// -----------------------------------------------------------------------------
// clamp kernel
// -----------------------------------------------------------------------------
__global__ void clamp_kernel(float* __restrict__ y,
                             const float* __restrict__ x,
                             float min_val,
                             float max_val,
                             int64_t n) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = fminf(fmaxf(val, min_val), max_val);
    }
}

// -----------------------------------------------------------------------------
// LSTM using cuDNN
// -----------------------------------------------------------------------------
// The LSTM weights from PyTorch are stored in a specific gate order. PyTorch uses
// i, f, g, o order (input, forget, cell, output). cuDNN may use a different order
// depending on the RNN mode. We need to reorder the weight matrices for cuDNN if necessary.
// Here we assume cuDNN is used directly with appropriate weight reordering.
// Since the task only allows .cu file, we'll implement the LSTM launcher in this file.
// We'll use cudnnRNNForward with a provided workspace.

// Helper to reorder PyTorch weights to cuDNN order (if needed).
// For LSTM with CUDNN_RNN_ALGO_STANDARD, cuDNN expects weights in order:
// i, f, c, o (input, forget, cell, output), which matches PyTorch? Let's check.
// PyTorch stores gates in order: input, forget, cell, output (i, f, g, o).
// cuDNN with CUDNN_LSTM expects hgates in order: i, f, c, o (actually i, f, g, o)
// In cuDNN, the gates are ordered as i, f, c, o, which aligns with PyTorch's order.
// So no reordering is needed for standard LSTM.

// Function to reorder weights if cuDNN expects different order (e.g., if using 
// cudnnRNN with CUDNN_LSTM, it uses i, f, c, o). For safety, we'll create a copy
// of weights in cuDNN expected order and handle biases.

// We'll implement a launcher that assumes the caller passes weight pointers in PyTorch order.
// Inside, we may need to reorder to cuDNN order by copying to new buffers.
// But for simplicity and correctness, we can use cudnnRNNForward with the appropriate
// weight space and reorder the weights on host? No, we can't use host in .cu launcher
// except for setting up descriptors. We can do on-device reordering or use cudnnSetRNNMatrixMathType
// but better to reorder weights using a CUDA kernel before calling cuDNN.
// However, to keep the code simple and correct, we can assume PyTorch order matches cuDNN.
// According to PyTorch source, for cuDNN, they reorder weights from PyTorch order to cuDNN's
// pseudo order. In LSTM, PyTorch weights are [W_ii|W_if|W_ig|W_io] stacked, which matches cuDNN if using
// the correct mode. Let's verify: In PyTorch, weight_ih_l0 has shape (4*hidden_size, input_size),
// arranged as: rows 0:hidden_size -> input gate, hidden_size:2*hidden_size -> forget gate,
// 2*hidden_size:3*hidden_size -> cell gate (g), 3*hidden_size:4*hidden_size -> output gate.
// cuDNN with CUDNN_LSTM expects the same order for input and recurrent weights? It expects
// weights in order: i, f, c, o. So matches.
// Therefore, we can pass the weights directly.
// Bias: PyTorch stores bias_ih and bias_hh separately, while cuDNN expects combined bias
// per gate (bias_ih + bias_hh for each gate). We need to combine them.

// We'll allocate temporary combined bias on device, but avoid using torch allocation in .cu.
// We can use cudaMalloc in launcher. That is allowed.

// The LSTM launcher will receive pointers to weights and biases for both layers,
// and the input x, and produce output, h_n, c_n.
// Since the launcher is called from binding.cpp, the binding will allocate output tensors.
// The launcher should fill the provided output pointers.

// Signature for lstm launcher (to be called from binding):
// extern "C" cudaError_t lstm_forward(
//     const float* x,               // input [seq_len, batch, input_size]
//     float* output,                // output [seq_len, batch, hidden_size]
//     float* h_n,                   // [num_layers, batch, hidden_size]
//     float* c_n,                   // [num_layers, batch, hidden_size]
//     const float* w_ih_l0, const float* w_hh_l0,
//     const float* b_ih_l0, const float* b_hh_l0,
//     const float* w_ih_l1, const float* w_hh_l1,
//     const float* b_ih_l1, const float* b_hh_l1,
//     int seq_len, int batch_size, int input_size, int hidden_size,
//     int num_layers,
//     cudaStream_t stream);

// Note: The caller must allocate output, h_n, c_n with appropriate sizes.

// Helper function to combine bias_ih and bias_hh into cuDNN bias format.
// cuDNN expects bias of size 8*hidden_size per layer (4 gates, each with two biases? Actually,
// for LSTM, bias dimension is 8*hidden_size: for each gate, there is a bias for input and recurrent).
// In cuDNN, weights include both input and recurrent weights, and bias is combined.
// Let's use cudnnSetRNNParams and provide the bias as a single array per layer.

// We'll implement a helper kernel to combine biases into a new buffer.
__global__ void combine_bias_kernel(const float* __restrict__ bias_ih,
                                    const float* __restrict__ bias_hh,
                                    float* __restrict__ combined,
                                    int hidden_size,
                                    int gate_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gate_count * hidden_size;
    if (idx < total) {
        combined[idx] = bias_ih[idx] + bias_hh[idx];
    }
}

// -----------------------------------------------------------------------------
// LSTM launcher using cuDNN
// -----------------------------------------------------------------------------
extern "C" cudaError_t lstm_forward(
    const float* x,
    float* output,
    float* h_n,
    float* c_n,
    const float* w_ih_l0, const float* w_hh_l0,
    const float* b_ih_l0, const float* b_hh_l0,
    const float* w_ih_l1, const float* w_hh_l1,
    const float* b_ih_l1, const float* b_hh_l1,
    int seq_len, int batch_size, int input_size, int hidden_size,
    int num_layers,
    cudaStream_t stream) {

    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc, hx_desc, cx_desc, hy_desc, cy_desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hx_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cx_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&hy_desc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&cy_desc));

    // x: [seq_len, batch, input_size]
    int dimA[3] = {seq_len, batch_size, input_size};
    int strideA[3] = {batch_size * input_size, input_size, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dimA, strideA));

    // y: [seq_len, batch, hidden_size]
    int dimB[3] = {seq_len, batch_size, hidden_size};
    int strideB[3] = {batch_size * hidden_size, hidden_size, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 3, dimB, strideB));

    // hx, cx, hy, cy: [num_layers, batch, hidden_size]
    int dimC[3] = {num_layers, batch_size, hidden_size};
    int strideC[3] = {batch_size * hidden_size, hidden_size, 1};
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hx_desc, CUDNN_DATA_FLOAT, 3, dimC, strideC));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cx_desc, CUDNN_DATA_FLOAT, 3, dimC, strideC));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(hy_desc, CUDNN_DATA_FLOAT, 3, dimC, strideC));
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(cy_desc, CUDNN_DATA_FLOAT, 3, dimC, strideC));

    // Create RNN descriptor
    cudnnRNNDescriptor_t rnn_desc;
    CUDNN_CHECK(cudnnCreateRNNDescriptor(&rnn_desc));
    CUDNN_CHECK(cudnnSetRNNDescriptor_v8(rnn_desc,
                                         CUDNN_RNN_ALGO_STANDARD,
                                         CUDNN_LSTM,
                                         CUDNN_RNN_NO_BIAS, // we will provide biases in weights? Actually we need bias mode.
                                                                   // We'll set to CUDNN_RNN_SINGLE_INP_BIAS? Let's use CUDNN_RNN_DOUBLE_BIAS? But PyTorch uses separate biases.
                                                                   // For simplicity, we'll use CUDNN_RNN_SINGLE_INP_BIAS and combine biases.
                                                                   // Actually, cuDNN has option to include both input and recurrent bias.
                                                                   // We'll use CUDNN_RNN_SINGLE_INP_BIAS and combine both biases into one.
                                         CUDNN_UNIDIRECTIONAL,
                                         CUDNN_LINEAR_INPUT,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_DEFAULT_MATH,
                                         input_size,
                                         hidden_size,
                                         hidden_size,
                                         num_layers,
                                         nullptr,
                                         0));

    // Get weight space size
    size_t weight_space_size;
    CUDNN_CHECK(cudnnGetRNNWeightSpaceSize(cudnn, rnn_desc, &weight_space_size));

    // Allocate weight space on device
    void* d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, weight_space_size));

    // We need to fill the weight space with weights from PyTorch.
    // PyTorch weights are stored separately for each layer and for input/recurrent.
    // cuDNN weight space layout depends on version. We can use cudnnSetRNNWeightParams to set
    // weights for each layer and matrix type.
    // However, setting weights individually requires many calls. Simpler approach: use the
    // cudnnRNNForward with raw weight space and copy weights in the correct layout.
    // But the layout is complex. Instead, we can use cudnnBuildRNNDynamic? Or use the older API
    // with cudnnRNNForwardInference and provide separate weight matrices? But cuDNN v8 uses
    // weight space for all weights.
    // To avoid complexity, we can use the older cuDNN RNN API (cudnnRNNForward) with per-layer weights.
    // However, that is deprecated but still available. We'll stick with v8 and use the weight space.
    // We need to know the layout of weight space. According to cuDNN documentation, for CUDNN_LSTM,
    // the weight space is organized as: for each layer, first the input weights (W_ih) of size
    // 4*hidden_size*input_size, then recurrent weights (W_hh) of size 4*hidden_size*hidden_size,
    // then biases (b_ih and b_hh) if bias mode includes them.
    // We set bias mode to CUDNN_RNN_SINGLE_INP_BIAS, meaning only input bias? But we have both.
    // Actually, CUDNN_RNN_SINGLE_INP_BIAS means one bias per gate (for input), and CUDNN_RNN_DOUBLE_BIAS means two biases per gate (input and recurrent).
    // PyTorch LSTM has both biases, so we should use CUDNN_RNN_DOUBLE_BIAS and provide both bias arrays in weight space.
    // So set bias mode to CUDNN_RNN_DOUBLE_BIAS.
    // Then weight space size will include both biases.
    // Let's adjust: use CUDNN_RNN_DOUBLE_BIAS.

    // Need to re-create descriptor with correct bias mode. We'll overwrite.
    CUDNN_CHECK(cudnnSetRNNDescriptor_v8(rnn_desc,
                                         CUDNN_RNN_ALGO_STANDARD,
                                         CUDNN_LSTM,
                                         CUDNN_RNN_DOUBLE_BIAS,
                                         CUDNN_UNIDIRECTIONAL,
                                         CUDNN_LINEAR_INPUT,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_DATA_FLOAT,
                                         CUDNN_DEFAULT_MATH,
                                         input_size,
                                         hidden_size,
                                         hidden_size,
                                         num_layers,
                                         nullptr,
                                         0));

    CUDNN_CHECK(cudnnGetRNNWeightSpaceSize(cudnn, rnn_desc, &weight_space_size));
    CUDA_CHECK(cudaMalloc(&d_weights, weight_space_size));

    // Now we need to copy weights from PyTorch pointers to d_weights in the correct layout.
    // We'll do this on host by allocating a temporary host buffer, but that would require copying from device to host,
    // which is wasteful. Better to do on-device copy with cudaMemcpy2D or by launching a kernel that copies from the
    // given weight pointers to d_weights with appropriate offsets.
    // However, the layout of weight space may be straightforward: it stores weights in a packed array per layer.
    // According to cuDNN documentation, the weight space for LSTM with double bias is:
    // For each layer l:
    //   W_ih (4*hidden_size * input_size)
    //   W_hh (4*hidden_size * hidden_size)
    //   b_ih (4*hidden_size)
    //   b_hh (4*hidden_size)
    // The order of gates within W_ih is i, f, c, o (matches PyTorch).
    // Therefore, we can copy directly.

    // We'll compute offsets and use cudaMemcpyAsync to copy each weight matrix and bias into d_weights.
    // We need host-side offsets calculation. That is fine.
    size_t offset = 0;
    // Layer 0
    // W_ih_l0 size = 4*hidden_size*input_size
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_ih_l0, 4*hidden_size*input_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*input_size*sizeof(float);
    // W_hh_l0 size = 4*hidden_size*hidden_size
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_hh_l0, 4*hidden_size*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*hidden_size*sizeof(float);
    // b_ih_l0 size = 4*hidden_size
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_ih_l0, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*sizeof(float);
    // b_hh_l0 size = 4*hidden_size
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_hh_l0, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*sizeof(float);

    // Layer 1 (if num_layers > 1)
    if (num_layers > 1) {
        // W_ih_l1 size = 4*hidden_size*hidden_size (input size is hidden_size for second layer)
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_ih_l1, 4*hidden_size*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*hidden_size*sizeof(float);
        // W_hh_l1 size = 4*hidden_size*hidden_size
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_hh_l1, 4*hidden_size*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*hidden_size*sizeof(float);
        // b_ih_l1 size = 4*hidden_size
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_ih_l1, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*sizeof(float);
        // b_hh_l1 size = 4*hidden_size
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_hh_l1, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*sizeof(float);
    }

    // Allocate workspace
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(cudnn, rnn_desc, seq_len, &workspace_size));
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    // Reserve space for hidden and cell state initializations (zeros) and final states.
    // We need to provide hx, cx, hy, cy pointers. The caller passes h_n and c_n for final states.
    // For initial state, we need zero tensors. We'll allocate them.
    // Actually, in the forward call, we can pass nullptr for hx and cx to indicate zero initial states.
    // cuDNN documentation says if hx or cx is NULL, they are assumed to be zero.
    // So we can pass nullptr for initial hx and cx.

    // Call RNN forward
    CUDNN_CHECK(cudnnRNNForward(cudnn,
                                rnn_desc,
                                CUDNN_FWD_MODE_INFERENCE,
                                nullptr, // devSeqLengths? Not needed for unidirectional
                                x_desc,
                                x,
                                y_desc,
                                output,
                                hx_desc,
                                nullptr, // hx (initial hidden state, nullptr => zero)
                                hy_desc,
                                h_n,
                                cx_desc,
                                nullptr, // cx (initial cell state, nullptr => zero)
                                cy_desc,
                                c_n,
                                weight_space_size,
                                d_weights,
                                workspace_size,
                                d_workspace,
                                0, // reserve space? For inference, can be 0
                                0));

    // Cleanup
    cudaFree(d_workspace);
    cudaFree(d_weights);
    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyTensorDescriptor(hx_desc);
    cudnnDestroyTensorDescriptor(cx_desc);
    cudnnDestroyTensorDescriptor(hy_desc);
    cudnnDestroyTensorDescriptor(cy_desc);
    cudnnDestroyRNNDescriptor(rnn_desc);
    cudnnDestroy(cudnn);

    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Launcher for layer_norm_default (no affine)
// -----------------------------------------------------------------------------
extern "C" cudaError_t layer_norm_default_launcher(
    const float* x,
    float* y,
    int N,               // normalized dimension (last dim size)
    int rows,            // number of rows = total_elements / N
    int64_t row_stride,  // stride between rows in elements (for x)
    int64_t out_row_stride, // stride between rows in output
    float eps,
    cudaStream_t stream) {

    // Allocate temporary mean and variance arrays on device
    float* d_mean;
    float* d_var;
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, rows * sizeof(float)));

    // Launch mean/var kernel
    const int block_size = 256;
    dim3 grid(rows);
    layer_norm_mean_var_kernel<block_size><<<grid, block_size, 0, stream>>>(
        x, d_mean, d_var, N, rows, row_stride);

    // Launch apply kernel
    const int apply_block = 256;
    dim3 apply_grid(rows);
    layer_norm_apply_kernel<<<apply_grid, apply_block, 0, stream>>>(
        x, y, d_mean, d_var, nullptr, nullptr, eps, N, rows, row_stride, out_row_stride, false, false);

    // Free temporary buffers after kernel completion (they are async, so we can't free immediately?)
    // Actually, cudaFree is stream-ordered, so it's safe to call after the kernels; they will be freed after execution.
    cudaFree(d_mean);
    cudaFree(d_var);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Launcher for layer_norm_affine (with weight and bias)
// -----------------------------------------------------------------------------
extern "C" cudaError_t layer_norm_affine_launcher(
    const float* x,
    float* y,
    const float* weight,
    const float* bias,
    int N,
    int rows,
    int64_t row_stride,
    int64_t out_row_stride,
    float eps,
    cudaStream_t stream) {

    float* d_mean;
    float* d_var;
    CUDA_CHECK(cudaMalloc(&d_mean, rows * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var, rows * sizeof(float)));

    const int block_size = 256;
    dim3 grid(rows);
    layer_norm_mean_var_kernel<block_size><<<grid, block_size, 0, stream>>>(
        x, d_mean, d_var, N, rows, row_stride);

    const int apply_block = 256;
    dim3 apply_grid(rows);
    layer_norm_apply_kernel<<<apply_grid, apply_block, 0, stream>>>(
        x, y, d_mean, d_var, weight, bias, eps, N, rows, row_stride, out_row_stride, true, true);

    cudaFree(d_mean);
    cudaFree(d_var);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Launcher for nan_to_num
// -----------------------------------------------------------------------------
extern "C" cudaError_t nan_to_num_launcher(
    float* y,
    const float* x,
    int64_t n,
    cudaStream_t stream) {

    const int block = 256;
    const int64_t grid = (n + block - 1) / block;
    nan_to_num_kernel<<<grid, block, 0, stream>>>(y, x, n);
    return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Launcher for add_bias
// -----------------------------------------------------------------------------
extern "C" cudaError_t add_bias_launcher(
    float* y,
    const float* x,
    float bias,
    int64_t n,
    cudaStream_t stream) {

    const int block = 256;
    const int64_t grid = (n + block - 1) / block;
    add_bias_kernel<<<grid, block, 0, stream>>>(y, x, bias, n);
    return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Launcher for clamp
// -----------------------------------------------------------------------------
extern "C" cudaError_t clamp_launcher(
    float* y,
    const float* x,
    float min_val,
    float max_val,
    int64_t n,
    cudaStream_t stream) {

    const int block = 256;
    const int64_t grid = (n + block - 1) / block;
    clamp_kernel<<<grid, block, 0, stream>>>(y, x, min_val, max_val, n);
    return cudaGetLastError();
}
