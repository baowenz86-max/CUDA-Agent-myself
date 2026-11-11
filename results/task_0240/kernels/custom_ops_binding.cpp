// Binding for bce_with_logits_forward
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void bce_with_logits_forward_launcher(
    float* output,
    const float* y_pred,
    const float* y_true,
    int64_t dim0,
    int64_t dim1,
    int64_t dim2,
    int64_t dim3,
    int64_t pred_stride_0,
    int64_t pred_stride_1,
    int64_t pred_stride_2,
    int64_t pred_stride_3,
    int64_t true_stride_0,
    int64_t true_stride_1,
    int64_t true_stride_2,
    int64_t true_stride_3,
    int64_t out_stride_0,
    int64_t out_stride_1,
    int64_t out_stride_2,
    int64_t out_stride_3,
    cudaStream_t stream);

torch::Tensor bce_with_logits_forward(torch::Tensor y_pred, torch::Tensor y_true) {
    // Input validation
    TORCH_CHECK(y_pred.is_cuda(), "y_pred must be a CUDA tensor");
    TORCH_CHECK(y_true.is_cuda(), "y_true must be a CUDA tensor");
    TORCH_CHECK(y_pred.dtype() == torch::kFloat32, "y_pred must be float32");
    TORCH_CHECK(y_true.dtype() == torch::kFloat32, "y_true must be float32");
    TORCH_CHECK(y_pred.sizes() == y_true.sizes(), "y_pred and y_true must have the same shape");
    TORCH_CHECK(y_pred.dim() == 4, "y_pred must be 4D");

    int64_t dim0 = y_pred.size(0);
    int64_t dim1 = y_pred.size(1);
    int64_t dim2 = y_pred.size(2);
    int64_t dim3 = y_pred.size(3);

    // Allocate output with same shape and strides as y_pred (contract says output shape same as y_pred)
    auto output = torch::empty_strided(y_pred.sizes(), y_pred.strides(), y_pred.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Get strides
    int64_t pred_stride_0 = y_pred.stride(0);
    int64_t pred_stride_1 = y_pred.stride(1);
    int64_t pred_stride_2 = y_pred.stride(2);
    int64_t pred_stride_3 = y_pred.stride(3);

    int64_t true_stride_0 = y_true.stride(0);
    int64_t true_stride_1 = y_true.stride(1);
    int64_t true_stride_2 = y_true.stride(2);
    int64_t true_stride_3 = y_true.stride(3);

    int64_t out_stride_0 = output.stride(0);
    int64_t out_stride_1 = output.stride(1);
    int64_t out_stride_2 = output.stride(2);
    int64_t out_stride_3 = output.stride(3);

    // Call launcher
    bce_with_logits_forward_launcher(
        output.data_ptr<float>(),
        y_pred.data_ptr<float>(),
        y_true.data_ptr<float>(),
        dim0, dim1, dim2, dim3,
        pred_stride_0, pred_stride_1, pred_stride_2, pred_stride_3,
        true_stride_0, true_stride_1, true_stride_2, true_stride_3,
        out_stride_0, out_stride_1, out_stride_2, out_stride_3,
        stream);

    return output;
}

void register_custom_ops_binding(pybind11::module& m) {
    m.def("bce_with_logits_forward", &bce_with_logits_forward,
          "Binary cross entropy with logits (elementwise, no reduction)",
          py::arg("y_pred"), py::arg("y_true"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
