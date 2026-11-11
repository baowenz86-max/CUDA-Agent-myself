#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void absolute_forward(
    float* output,
    const float* input,
    int64_t numel,
    cudaStream_t stream);

extern "C" void sync_batch_norm_eval_forward(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int64_t numel,
    int64_t C,
    int64_t HW,
    cudaStream_t stream);

// absolute: element-wise absolute value
torch::Tensor absolute(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    // x can be non-contiguous due to strides; we require it to be at least not overlapping.
    // The launcher uses raw pointer, so we make it contiguous to avoid stride issues.
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    absolute_forward(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        x_contig.numel(),
        stream);

    // If original x was non-contiguous, output should match original strides? 
    // Contract says output same as x, but if we use contiguous x then output stride is contiguous,
    // which may differ from original x stride. But original x stride is standard contiguous for this task.
    // For safety, we return output with same shape and contiguous strides; if original x was non-contiguous,
    // we would need to make output contiguous and then perhaps reshape. Since contract says "supports strides"
    // we should preserve input stride. But launcher works on contiguous memory. To preserve stride:
    // We can create output with same options as x (including strides) and then copy data.
    // However, the launcher writes to output pointer assuming contiguous layout.
    // Since input x is likely contiguous in this task (given shape and stride from contract),
    // we assume contiguity after check; if not, we make contiguous and then return contiguous output.
    // To be strict, we should preserve original stride. But model_new uses x directly after absolute,
    // and later sync_batch_norm_eval uses x, which expects contiguous anyway. We'll keep this simple:
    // If x is not contiguous, we return a contiguous tensor; otherwise return output with same stride as x.
    // Since task requires shape and dtype match, stride is less critical but mentioned in contract returns.
    // Actually returns shape same as x, not necessarily stride. So contiguous output is acceptable.
    return output;
}

// sync_batch_norm_eval: inference batch norm using running stats
torch::Tensor sync_batch_norm_eval(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kFloat32, "weight must be CUDA float32");
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32, "bias must be CUDA float32");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32, "running_mean must be CUDA float32");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32, "running_var must be CUDA float32");
    TORCH_CHECK(weight.dim() == 1 && bias.dim() == 1 && running_mean.dim() == 1 && running_var.dim() == 1,
                "Normalization parameters must be 1D");
    int64_t C = x.size(1);
    TORCH_CHECK(weight.numel() == C && bias.numel() == C && running_mean.numel() == C && running_var.numel() == C,
                "Parameter size must match channel count");
    // Make x contiguous for kernel access
    auto x_contig = x.contiguous();
    auto output = torch::empty_like(x_contig);
    int64_t HW = x_contig.size(2) * x_contig.size(3);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    sync_batch_norm_eval_forward(
        output.data_ptr<float>(),
        x_contig.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        x_contig.numel(),
        C,
        HW,
        stream);

    return output;
}

// Register functions
void register_custom_ops(pybind11::module& m) {
    m.def("absolute", &absolute, "Element-wise absolute value",
          py::arg("x"));
    m.def("sync_batch_norm_eval", &sync_batch_norm_eval, "SyncBatchNorm eval mode",
          py::arg("x"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
