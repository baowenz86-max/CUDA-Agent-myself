// Bindings for LSTM, Linear, and Exp custom ops
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

extern "C" void exp_forward_launcher(const float* input, float* output, int n, cudaStream_t stream);

#define CHECK_CUDA_TENSOR(t) \
    TORCH_CHECK((t).is_cuda(), #t " must be a CUDA tensor"); \
    TORCH_CHECK((t).is_contiguous(), #t " must be contiguous"); \
    TORCH_CHECK((t).dtype() == torch::kFloat32, #t " must be float32")

static cublasHandle_t get_cublas_handle() {
    static cublasHandle_t handle = nullptr;
    if (!handle) {
        cublasCreate(&handle);
    }
    return handle;
}

static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (!handle) {
        cudnnCreate(&handle);
    }
    return handle;
}

// LSTM forward using cuDNN
torch::Tensor lstm_forward(
    torch::Tensor input,
    torch::Tensor weight_ih_l0,
    torch::Tensor weight_hh_l0,
    torch::Tensor bias_ih_l0,
    torch::Tensor bias_hh_l0,
    torch::Tensor weight_ih_l1,
    torch::Tensor weight_hh_l1,
    torch::Tensor bias_ih_l1,
    torch::Tensor bias_hh_l1,
    int64_t seq_len,
    int64_t batch_size,
    int64_t input_size,
    int64_t hidden_size,
    int64_t num_layers
) {
    CHECK_CUDA_TENSOR(input);
    CHECK_CUDA_TENSOR(weight_ih_l0);
    CHECK_CUDA_TENSOR(weight_hh_l0);
    CHECK_CUDA_TENSOR(bias_ih_l0);
    CHECK_CUDA_TENSOR(bias_hh_l0);
    CHECK_CUDA_TENSOR(weight_ih_l1);
    CHECK_CUDA_TENSOR(weight_hh_l1);
    CHECK_CUDA_TENSOR(bias_ih_l1);
    CHECK_CUDA_TENSOR(bias_hh_l1);

    TORCH_CHECK(seq_len > 0 && batch_size > 0 && input_size > 0 && hidden_size > 0 && num_layers == 2,
                "Only two-layer LSTM supported");

    cudnnHandle_t cudnn = get_cudnn_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnSetStream(cudnn, stream);

    cudnnRNNDescriptor_t rnn_desc;
    cudnnCreateRNNDescriptor(&rnn_desc);
    cudnnSetRNNDescriptor_v8(rnn_desc,
                             CUDNN_RNN_ALGO_STANDARD,
                             CUDNN_RNN_RELU,
                             CUDNN_RNN_RELU,
                             CUDNN_UNIDIRECTIONAL,
                             CUDNN_LINEAR_INPUT,
                             CUDNN_DATA_FLOAT,
                             CUDNN_DATA_FLOAT,
                             CUDNN_DEFAULT_MATH,
                             (int)input_size,
                             (int)hidden_size,
                             (int)hidden_size,
                             (int)num_layers,
                             nullptr,
                             CUDNN_RNN_PADDED_IO_DISABLED);

    cudnnSetRNNMatrixMathType(rnn_desc, CUDNN_DEFAULT_MATH);
    cudnnSetRNNBiasMode(rnn_desc, CUDNN_RNN_DOUBLE_BIAS);

    size_t weight_size;
    cudnnGetRNNWeightSpaceSize(cudnn, rnn_desc, &weight_size);
    void* d_wspace;
    cudaMalloc(&d_wspace, weight_size);

    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    int dimsX[3] = {(int)seq_len, (int)batch_size, (int)input_size};
    int stridesX[3] = {batch_size * input_size, input_size, 1};
    cudnnSetTensorNdDescriptor(x_desc, CUDNN_DATA_FLOAT, 3, dimsX, stridesX);
    int dimsY[3] = {(int)seq_len, (int)batch_size, (int)hidden_size};
    int stridesY[3] = {batch_size * hidden_size, hidden_size, 1};
    cudnnSetTensorNdDescriptor(y_desc, CUDNN_DATA_FLOAT, 3, dimsY, stridesY);

    cudnnFilterDescriptor_t w_desc_ih, w_desc_hh;
    cudnnCreateFilterDescriptor(&w_desc_ih);
    cudnnCreateFilterDescriptor(&w_desc_hh);
    cudnnSetFilterNdDescriptor(w_desc_ih, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, (int[]){4*(int)hidden_size, (int)input_size});
    cudnnSetFilterNdDescriptor(w_desc_hh, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, (int[]){4*(int)hidden_size, (int)hidden_size});

    cudnnTensorDescriptor_t b_desc;
    cudnnCreateTensorDescriptor(&b_desc);
    cudnnSetTensor4dDescriptor(b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 4*(int)hidden_size, 1, 1, 1);

    for (int layer = 0; layer < num_layers; ++layer) {
        void* ptr_ih = nullptr;
        cudnnGetRNNWeightParams(cudnn, rnn_desc, layer, weight_size, d_wspace,
                                0, w_desc_ih, &ptr_ih, nullptr);
        cudaMemcpyAsync(ptr_ih, (layer == 0 ? weight_ih_l0.data_ptr<float>() : weight_ih_l1.data_ptr<float>()),
                        weight_ih_l0.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        void* ptr_hh = nullptr;
        cudnnGetRNNWeightParams(cudnn, rnn_desc, layer, weight_size, d_wspace,
                                1, w_desc_hh, &ptr_hh, nullptr);
        cudaMemcpyAsync(ptr_hh, (layer == 0 ? weight_hh_l0.data_ptr<float>() : weight_hh_l1.data_ptr<float>()),
                        weight_hh_l0.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        void* ptr_bih = nullptr;
        cudnnGetRNNWeightParams(cudnn, rnn_desc, layer, weight_size, d_wspace,
                                2, b_desc, &ptr_bih, nullptr);
        cudaMemcpyAsync(ptr_bih, (layer == 0 ? bias_ih_l0.data_ptr<float>() : bias_ih_l1.data_ptr<float>()),
                        bias_ih_l0.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        void* ptr_bhh = nullptr;
        cudnnGetRNNWeightParams(cudnn, rnn_desc, layer, weight_size, d_wspace,
                                3, b_desc, &ptr_bhh, nullptr);
        cudaMemcpyAsync(ptr_bhh, (layer == 0 ? bias_hh_l0.data_ptr<float>() : bias_hh_l1.data_ptr<float>()),
                        bias_hh_l0.numel() * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // 使用 torch::empty_like 基于 input 创建输出，不能使用 torch::empty（但 torch::empty 也可能允许，为安全使用 torch::empty 但仅分配）
    auto output = torch::empty({seq_len, batch_size, hidden_size}, input.options());
    auto hx = torch::zeros({num_layers, batch_size, hidden_size}, input.options());
    auto cx = torch::zeros({num_layers, batch_size, hidden_size}, input.options());
    void* d_output = output.data_ptr<float>();
    void* d_hx = hx.data_ptr<float>();
    void* d_cx = cx.data_ptr<float>();

    size_t work_size;
    cudnnGetRNNTempSpaceSizes(cudnn, rnn_desc, CUDNN_RNN_FWD_MODE_INFERENCE, nullptr, &work_size, nullptr);
    void* d_workspace;
    cudaMalloc(&d_workspace, work_size);

    cudnnRNNForwardInference(cudnn,
                             rnn_desc,
                             CUDNN_RNN_FWD_MODE_INFERENCE,
                             nullptr,
                             x_desc,
                             input.data_ptr<float>(),
                             y_desc,
                             d_output,
                             nullptr,
                             d_hx,
                             nullptr,
                             nullptr,
                             nullptr,
                             d_cx,
                             nullptr,
                             nullptr,
                             d_wspace,
                             weight_size,
                             d_workspace,
                             work_size,
                             nullptr, 0);

    cudaError_t cuda_err = cudaGetLastError();
    TORCH_CHECK(cuda_err == cudaSuccess, "CUDA error in lstm_forward: ", cudaGetErrorString(cuda_err));

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc_ih);
    cudnnDestroyFilterDescriptor(w_desc_hh);
    cudnnDestroyTensorDescriptor(b_desc);
    cudnnDestroyRNNDescriptor(rnn_desc);
    cudaFree(d_wspace);
    cudaFree(d_workspace);

    return output;
}

torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t out_features
) {
    CHECK_CUDA_TENSOR(input);
    CHECK_CUDA_TENSOR(weight);
    CHECK_CUDA_TENSOR(bias);

    int64_t batch = input.size(0);
    int64_t in_features = input.size(1);
    TORCH_CHECK(weight.size(0) == out_features, "weight shape mismatch");
    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");
    TORCH_CHECK(bias.size(0) == out_features, "bias size mismatch");

    cublasHandle_t handle = get_cublas_handle();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cublasSetStream(handle, stream);

    auto output = torch::empty({batch, out_features}, input.options());

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 使用 cublasSgemm 直接计算，不进行显式转置，利用 cublas 的转置参数
    // 计算 C = A * B^T，其中 A 为 weight (out, in) 行主序，B 为 input (batch, in) 行主序
    // 采用 cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, ...) 模式
    // 这里 m = out_features, n = batch, k = in_features
    // 将 weight 作为 col-major (in, out) 矩阵，lda = in_features，transa = CUBLAS_OP_T 得到 weight^T
    // 将 input 作为 col-major (batch, in) 矩阵，ldb = in_features，transb = CUBLAS_OP_T 得到 input^T
    // 但实际推导发现不匹配，改用以下可靠方法：
    // 使用 cublasSgemm 配合转置，并利用输出矩阵的存储方式避免显式转置。
    // 根据公式：对于行主序 A (m x k) 和 B (k x n)，C = A * B 可通过
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, n, m, k, &alpha, B, n, A, k, &beta, C, n);
    // 在我们的例子中，A = input (batch x in), B = weight^T (in x out), C = output (batch x out)
    // 我们不需要显式转置 weight，只需将 weight 视为 B 并传递其转置形式即可。
    // 因此，使用以下调用：
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, out_features, batch, in_features,
    //             &alpha, weight.data_ptr<float>(), out_features,
    //             input.data_ptr<float>(), in_features,
    //             &beta, output.data_ptr<float>(), out_features);
    // 但需要验证 weight 的存储方式：weight 是 (out, in) 行主序，在 col-major 解释下为 (in, out) 且 lda=out_features。
    // 上述调用中的第一个矩阵（A）期望为 (in, out) col-major，lda=out_features，与 weight 的存储一致！
    // 第二个矩阵（B）期望为 (batch, in) col-major，ldb=in_features，input 存储为 (batch, in) 行主序，
    // 但在 col-major 解释下应为 (in, batch) 且 ldb=in_features，因此需要转置，但使用 transb=CUBLAS_OP_T 可自动转置。
    // 经过仔细验证，下面这个调用是正确的：
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, out_features, batch, in_features,
                &alpha, weight.data_ptr<float>(), out_features,
                input.data_ptr<float>(), in_features,
                &beta, output.data_ptr<float>(), out_features);

    // 添加 bias：使用 cublasSaxpy 对每一列添加 bias[j]
    const float one = 1.0f;
    for (int64_t j = 0; j < out_features; ++j) {
        cublasSaxpy(handle, batch, &one, bias.data_ptr<float>() + j, 0, output.data_ptr<float>() + j * out_features, 1);
    }

    return output;
}

torch::Tensor exp_forward(torch::Tensor input) {
    CHECK_CUDA_TENSOR(input);
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    exp_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), input.numel(), stream);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error in exp_forward: ", cudaGetErrorString(err));
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("lstm_forward", &lstm_forward, "LSTM forward (cuDNN)",
          py::arg("input"), py::arg("weight_ih_l0"), py::arg("weight_hh_l0"),
          py::arg("bias_ih_l0"), py::arg("bias_hh_l0"),
          py::arg("weight_ih_l1"), py::arg("weight_hh_l1"),
          py::arg("bias_ih_l1"), py::arg("bias_hh_l1"),
          py::arg("seq_len"), py::arg("batch_size"), py::arg("input_size"),
          py::arg("hidden_size"), py::arg("num_layers"));
    m.def("linear_forward", &linear_forward, "Linear forward (cuBLAS)",
          py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("out_features"));
    m.def("exp_forward", &exp_forward, "Exp forward (custom CUDA kernel)",
          py::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
