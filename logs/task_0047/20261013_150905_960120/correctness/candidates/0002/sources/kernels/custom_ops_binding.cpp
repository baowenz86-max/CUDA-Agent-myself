#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void linear_forward_cuda(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    int input_stride_row,
    int input_stride_col,
    int weight_stride_row,
    int weight_stride_col,
    int output_stride_row,
    int output_stride_col,
    cudaStream_t stream
);

extern "C" void softmax_forward_cuda(
    const float* input,
    float* output,
    int num_rows,
    int num_cols,
    int in_stride_row,
    int in_stride_col,
    int out_stride_row,
    int out_stride_col,
    cudaStream_t stream
);

extern "C" void log_softmax_forward_cuda(
    const float* input,
    float* output,
    int num_rows,
    int num_cols,
    int in_stride_row,
    int in_stride_col,
    int out_stride_row,
    int out_stride_col,
    cudaStream_t stream
);

extern "C" void kl_div_batchmean_forward_cuda(
    const float* input,
    const float* target,
    float* output,
    int num_rows,
    int num_cols,
    int in_stride_row,
    int in_stride_col,
    int target_stride_row,
    int target_stride_col,
    cudaStream_t stream
);

extern "C" void l1_loss_sum_forward_cuda(
    const float* x,
    const float* params,
    float* output,
    int batch_size,
    int num_params,
    int x_stride_row,
    int x_stride_col,
    cudaStream_t stream
);

// Helper to check CUDA tensor and dtype
static void check_cuda_float(const torch::Tensor& t, const std::string& name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

// linear_forward: input [batch, in], weight [out, in], bias [out] -> output [batch, out]
torch::Tensor linear_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias) {
    check_cuda_float(input, "input");
    check_cuda_float(weight, "weight");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "weight must be 2D");
    int64_t batch_size = input.size(0);
    int64_t in_features = input.size(1);
    int64_t out_features = weight.size(0);
    TORCH_CHECK(weight.size(1) == in_features, "weight shape mismatch");
    if (bias.defined()) {
        check_cuda_float(bias, "bias");
        TORCH_CHECK(bias.dim() == 1 && bias.size(0) == out_features, "bias shape mismatch");
    }

    // Ensure contiguous tensors for cuBLAS and bias add
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto output = torch::empty({batch_size, out_features}, input.options());
    auto output_contig = output; // already contiguous

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const float* bias_ptr = bias.defined() ? bias.contiguous().data_ptr<float>() : nullptr;

    linear_forward_cuda(
        input_contig.data_ptr<float>(),
        weight_contig.data_ptr<float>(),
        bias_ptr,
        output_contig.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(in_features),
        static_cast<int>(out_features),
        0, // input_stride_row (unused in contiguous case, but pass 0; launcher ignores if contiguous)
        0,
        0,
        0,
        0,
        0,
        stream
    );
    return output;
}

// softmax_forward: input [batch, classes] -> output same shape
torch::Tensor softmax_forward(torch::Tensor input, int64_t dim) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(dim == 1, "only dim=1 supported");
    auto output = torch::empty_like(input);
    auto input_contig = input.contiguous();
    auto output_contig = output; // already contiguous
    int64_t rows = input.size(0);
    int64_t cols = input.size(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    softmax_forward_cuda(
        input_contig.data_ptr<float>(),
        output_contig.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols), // in_stride_row = cols
        1,                      // in_stride_col
        static_cast<int>(cols), // out_stride_row
        1,                      // out_stride_col
        stream
    );
    return output;
}

// log_softmax_forward: input [batch, classes] -> output same shape
torch::Tensor log_softmax_forward(torch::Tensor input, int64_t dim) {
    check_cuda_float(input, "input");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(dim == 1, "only dim=1 supported");
    auto output = torch::empty_like(input);
    auto input_contig = input.contiguous();
    auto output_contig = output;
    int64_t rows = input.size(0);
    int64_t cols = input.size(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    log_softmax_forward_cuda(
        input_contig.data_ptr<float>(),
        output_contig.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols),
        1,
        static_cast<int>(cols),
        1,
        stream
    );
    return output;
}

// kl_div_batchmean_forward: input [batch, classes], target [batch, classes] -> scalar
torch::Tensor kl_div_batchmean_forward(torch::Tensor input, torch::Tensor target) {
    check_cuda_float(input, "input");
    check_cuda_float(target, "target");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(target.dim() == 2, "target must be 2D");
    TORCH_CHECK(input.sizes() == target.sizes(), "input and target must have same shape");
    auto output = torch::empty({}, input.options());
    auto input_contig = input.contiguous();
    auto target_contig = target.contiguous();
    int64_t rows = input.size(0);
    int64_t cols = input.size(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    kl_div_batchmean_forward_cuda(
        input_contig.data_ptr<float>(),
        target_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(rows),
        static_cast<int>(cols),
        static_cast<int>(cols),
        1,
        static_cast<int>(cols),
        1,
        stream
    );
    return output;
}

// l1_loss_sum_forward: x [batch, num_params], params list of scalar tensors -> scalar
// params is given as a Python list of tensors; we flatten into a contiguous 1D tensor.
torch::Tensor l1_loss_sum_forward(torch::Tensor x, std::vector<torch::Tensor> params) {
    check_cuda_float(x, "x");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    int64_t batch_size = x.size(0);
    int64_t num_params = x.size(1);
    TORCH_CHECK(static_cast<int64_t>(params.size()) == num_params, "params count mismatch");
    // Create contiguous 1D params tensor on same device
    auto params_tensor = torch::empty({num_params}, x.options());
    for (int64_t i = 0; i < num_params; ++i) {
        auto p = params[i];
        TORCH_CHECK(p.is_cuda() && p.dtype() == torch::kFloat32, "param must be float32 CUDA");
        TORCH_CHECK(p.numel() == 1, "each param must be scalar");
        params_tensor[i] = p;
    }
    auto output = torch::empty({}, x.options());
    auto x_contig = x.contiguous();
    auto params_contig = params_tensor.contiguous();
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    l1_loss_sum_forward_cuda(
        x_contig.data_ptr<float>(),
        params_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(batch_size),
        static_cast<int>(num_params),
        static_cast<int>(num_params), // x_stride_row
        1,                            // x_stride_col
        stream
    );
    return output;
}

void register_custom_ops(pybind11::module& m) {
    m.def("linear_forward", &linear_forward, "Linear forward (input @ weight.T + bias)",
          py::arg("input"), py::arg("weight"), py::arg("bias"));
    m.def("softmax_forward", &softmax_forward, "Softmax along dim=1",
          py::arg("input"), py::arg("dim") = 1);
    m.def("log_softmax_forward", &log_softmax_forward, "Log softmax along dim=1",
          py::arg("input"), py::arg("dim") = 1);
    m.def("kl_div_batchmean_forward", &kl_div_batchmean_forward, "KL divergence batchmean",
          py::arg("input"), py::arg("target"));
    m.def("l1_loss_sum_forward", &l1_loss_sum_forward, "L1 loss sum over params",
          py::arg("x"), py::arg("params"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
