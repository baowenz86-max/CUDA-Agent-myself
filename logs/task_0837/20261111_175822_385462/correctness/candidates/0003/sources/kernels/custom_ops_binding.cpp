#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void adaptive_logsoftmax_head_launcher(
    const float* x, int x_rows, int x_cols, int x_row_stride,
    const float* head_weight, int head_rows, int head_cols, int head_row_stride,
    float* output, int out_rows, int out_cols, int out_row_stride,
    cudaStream_t stream);

extern "C" void adaptive_logsoftmax_tail_launcher(
    const float* x, int x_rows, int x_cols, int x_row_stride,
    const float* tail0_weight, int tail0_rows, int tail0_cols, int tail0_row_stride,
    const float* tail1_weight, int tail1_rows, int tail1_cols, int tail1_row_stride,
    float* output, int out_rows, int out_cols, int out_row_stride,
    cudaStream_t stream);

extern "C" void adaptive_logsoftmax_combine_launcher(
    const float* head, int head_rows, int head_total_cols, int head_row_stride,
    const float* tail, int tail_rows, int tail_cols, int tail_row_stride,
    float* output, int out_d0, int out_d1, int out_d2, int out_row_stride, int out_col_stride,
    int head_copy_cols,
    cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Check tensor is CUDA float32 and optionally contiguous
torch::Tensor adaptive_logsoftmax_head_forward(torch::Tensor x, torch::Tensor head_weight) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(head_weight.is_cuda(), "head_weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(head_weight.scalar_type() == torch::kFloat32, "head_weight must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(head_weight.dim() == 2, "head_weight must be 2D");
    TORCH_CHECK(x.size(1) == head_weight.size(1), "inner dimensions mismatch");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    // Use the full head weight rows (5001) for log-softmax
    int n_head_classes = head_weight.size(0);
    TORCH_CHECK(n_head_classes == 5001, "head_weight must have 5001 rows");

    auto output = torch::empty({batch_size, n_head_classes}, x.options());

    auto x_contig = x.contiguous();
    auto head_contig = head_weight.contiguous();

    cudaStream_t stream = getCurrentStream();
    adaptive_logsoftmax_head_launcher(
        x_contig.data_ptr<float>(), batch_size, in_features, x_contig.stride(0),
        head_contig.data_ptr<float>(), n_head_classes, in_features, head_contig.stride(0),
        output.data_ptr<float>(), batch_size, n_head_classes, output.stride(0),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor adaptive_logsoftmax_tail_forward(torch::Tensor x, torch::Tensor tail_0_weight, torch::Tensor tail_1_weight) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(tail_0_weight.is_cuda(), "tail_0_weight must be a CUDA tensor");
    TORCH_CHECK(tail_1_weight.is_cuda(), "tail_1_weight must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(tail_0_weight.scalar_type() == torch::kFloat32, "tail_0_weight must be float32");
    TORCH_CHECK(tail_1_weight.scalar_type() == torch::kFloat32, "tail_1_weight must be float32");
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(tail_0_weight.dim() == 2, "tail_0_weight must be 2D");
    TORCH_CHECK(tail_1_weight.dim() == 2, "tail_1_weight must be 2D");
    TORCH_CHECK(x.size(1) == tail_0_weight.size(1), "x and tail_0_weight inner dims must match");
    TORCH_CHECK(tail_0_weight.size(0) == tail_1_weight.size(1), "tail_0 output and tail_1 input dims must match");

    int batch_size = x.size(0);
    int in_features = x.size(1);
    int hidden_size = tail_0_weight.size(0);
    int n_tail_classes = tail_1_weight.size(0);

    auto output = torch::empty({batch_size, n_tail_classes}, x.options());

    auto x_contig = x.contiguous();
    auto tail_0_contig = tail_0_weight.contiguous();
    auto tail_1_contig = tail_1_weight.contiguous();

    cudaStream_t stream = getCurrentStream();
    adaptive_logsoftmax_tail_launcher(
        x_contig.data_ptr<float>(), batch_size, in_features, x_contig.stride(0),
        tail_0_contig.data_ptr<float>(), hidden_size, in_features, tail_0_contig.stride(0),
        tail_1_contig.data_ptr<float>(), n_tail_classes, hidden_size, tail_1_contig.stride(0),
        output.data_ptr<float>(), batch_size, n_tail_classes, output.stride(0),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

torch::Tensor adaptive_logsoftmax_combine(torch::Tensor head_log_probs, torch::Tensor tail_log_probs) {
    TORCH_CHECK(head_log_probs.is_cuda(), "head_log_probs must be a CUDA tensor");
    TORCH_CHECK(tail_log_probs.is_cuda(), "tail_log_probs must be a CUDA tensor");
    TORCH_CHECK(head_log_probs.scalar_type() == torch::kFloat32, "head_log_probs must be float32");
    TORCH_CHECK(tail_log_probs.scalar_type() == torch::kFloat32, "tail_log_probs must be float32");
    TORCH_CHECK(head_log_probs.dim() == 2, "head_log_probs must be 2D");
    TORCH_CHECK(tail_log_probs.dim() == 2, "tail_log_probs must be 2D");
    TORCH_CHECK(head_log_probs.size(0) == tail_log_probs.size(0), "batch sizes must match");

    int batch_size = head_log_probs.size(0);
    int head_total_cols = head_log_probs.size(1);  // expected 5001
    int tail_cols = tail_log_probs.size(1);       // expected 5000
    TORCH_CHECK(head_total_cols == 5001, "head_log_probs must have 5001 columns");
    TORCH_CHECK(tail_cols == 5000, "tail_log_probs must have 5000 columns");
    int head_copy_cols = 5000; // only first 5000 head columns are used
    int n_classes = head_copy_cols + tail_cols; // 10000

    auto output = torch::empty({batch_size, 1, n_classes}, head_log_probs.options());

    auto head_contig = head_log_probs.contiguous();
    auto tail_contig = tail_log_probs.contiguous();

    int out_row_stride = output.stride(0);
    int out_col_stride = output.stride(2);

    cudaStream_t stream = getCurrentStream();
    adaptive_logsoftmax_combine_launcher(
        head_contig.data_ptr<float>(), batch_size, head_total_cols, head_contig.stride(0),
        tail_contig.data_ptr<float>(), batch_size, tail_cols, tail_contig.stride(0),
        output.data_ptr<float>(), batch_size, 1, n_classes, out_row_stride, out_col_stride,
        head_copy_cols,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

void register_adaptive_logsoftmax_ops(pybind11::module& m) {
    m.def("adaptive_logsoftmax_head_forward", &adaptive_logsoftmax_head_forward,
          "Compute head log softmax over 5001 classes",
          py::arg("x"), py::arg("head_weight"));
    m.def("adaptive_logsoftmax_tail_forward", &adaptive_logsoftmax_tail_forward,
          "Compute tail log softmax",
          py::arg("x"), py::arg("tail_0_weight"), py::arg("tail_1_weight"));
    m.def("adaptive_logsoftmax_combine", &adaptive_logsoftmax_combine,
          "Combine first 5000 head log probs with tail log probs",
          py::arg("head_log_probs"), py::arg("tail_log_probs"));
}

REGISTER_BINDING(adaptive_logsoftmax_ops, register_adaptive_logsoftmax_ops);
