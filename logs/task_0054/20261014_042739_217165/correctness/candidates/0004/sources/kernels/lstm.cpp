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

// LSTM using cuDNN
// The LSTM weights from PyTorch are stored in a specific gate order. PyTorch uses
// i, f, g, o order (input, forget, cell, output). cuDNN uses the same order (i, f, c, o)
// for CUDNN_LSTM, so no reordering is needed.
// Bias: PyTorch stores bias_ih and bias_hh separately, while cuDNN expects separate bias
// arrays in its weight space when using CUDNN_RNN_DOUBLE_BIAS mode.
extern "C" cudaError_t lstm_forward(
    const float* x,               // input [seq_len, batch, input_size]
    float* output,                // output [seq_len, batch, hidden_size]
    float* h_n,                   // [num_layers, batch, hidden_size]
    float* c_n,                   // [num_layers, batch, hidden_size]
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

    // Get weight space size
    size_t weight_space_size;
    CUDNN_CHECK(cudnnGetRNNWeightSpaceSize(cudnn, rnn_desc, &weight_space_size));

    // Allocate weight space on device
    void* d_weights;
    CUDA_CHECK(cudaMalloc(&d_weights, weight_space_size));

    // Copy weights and biases into weight space.
    // cuDNN weight space layout for LSTM with double bias:
    // For each layer l:
    //   W_ih (4*hidden_size * input_size)
    //   W_hh (4*hidden_size * hidden_size)
    //   b_ih (4*hidden_size)
    //   b_hh (4*hidden_size)
    // The order of gates within W_ih is i, f, c, o (matches PyTorch).
    size_t offset = 0;
    // Layer 0
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_ih_l0, 4*hidden_size*input_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*input_size*sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_hh_l0, 4*hidden_size*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*hidden_size*sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_ih_l0, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_hh_l0, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
    offset += 4*hidden_size*sizeof(float);

    // Layer 1 (if num_layers > 1)
    if (num_layers > 1) {
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_ih_l1, 4*hidden_size*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*hidden_size*sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, w_hh_l1, 4*hidden_size*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*hidden_size*sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_ih_l1, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync((char*)d_weights + offset, b_hh_l1, 4*hidden_size*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        offset += 4*hidden_size*sizeof(float);
    }

    // Allocate workspace
    size_t workspace_size;
    CUDNN_CHECK(cudnnGetRNNWorkspaceSize(cudnn, rnn_desc, seq_len, &workspace_size));
    void* d_workspace;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));

    // Initial hidden and cell states are zero (pass nullptr)
    // Call RNN forward
    CUDNN_CHECK(cudnnRNNForward(cudnn,
                                rnn_desc,
                                CUDNN_FWD_MODE_INFERENCE,
                                nullptr, // devSeqLengths not needed for unidirectional
                                x_desc,
                                x,
                                y_desc,
                                output,
                                hx_desc,
                                nullptr, // hx (initial hidden state, zero)
                                hy_desc,
                                h_n,
                                cx_desc,
                                nullptr, // cx (initial cell state, zero)
                                cy_desc,
                                c_n,
                                weight_space_size,
                                d_weights,
                                workspace_size,
                                d_workspace,
                                0, // reserve space
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
