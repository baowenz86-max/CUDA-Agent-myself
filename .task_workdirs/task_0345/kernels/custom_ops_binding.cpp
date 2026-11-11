// Bindings for custom unfold and batch_norm operations
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void op_unfold_launcher(const float* x,
                                   float* out,
                                   int N, int C, int H, int W,
                                   int kernel_h, int kernel_w,
                                   int dilation_h, int dilation_w,
                                   cudaStream_t stream);

extern "C" void op_batch_norm_1d_training_launcher(const float* input,
                                                   float* output,
                                                   const float* weight,
                                                   const float* bias,
                                                   float* running_mean,
                                                   float* running_var,
                                                   int N, int C, int L,
                                                   float eps, float momentum,
                                                   cudaStream_t stream);

// Wrapper for unfold
torch::Tensor op_unfold_forward(torch::Tensor x,
                                int64_t kernel_h,
                                int64_t kernel_w,
                                int64_t dilation_h,
                                int64_t dilation_w) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    int kh = static_cast<int>(kernel_h);
    int kw = static_cast<int>(kernel_w);
    int dh = static_cast<int>(dilation_h);
    int dw = static_cast<int>(dilation_w);

    TORCH_CHECK(kh > 0 && kw > 0, "kernel sizes must be positive");
    TORCH_CHECK(dh > 0 && dw > 0, "dilations must be positive");

    int out_H = (H - dh * (kh - 1) - 1) + 1;
    int out_W = (W - dw * (kw - 1) - 1) + 1;
    TORCH_CHECK(out_H > 0 && out_W > 0, "Invalid output dimensions");
    int L = out_H * out_W;
    int out_C = C * kh * kw;

    auto output = torch::empty({N, out_C, L}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    op_unfold_launcher(x.data_ptr<float>(),
                       output.data_ptr<float>(),
                       N, C, H, W,
                       kh, kw,
                       dh, dw,
                       stream);

    return output;
}

// Wrapper for batch_norm_1d_training
torch::Tensor op_batch_norm_1d_training_forward(torch::Tensor input,
                                                c10::optional<torch::Tensor> weight,
                                                c10::optional<torch::Tensor> bias,
                                                torch::Tensor running_mean,
                                                torch::Tensor running_var,
                                                double eps,
                                                double momentum) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (N,C,L)");

    int N = input.size(0);
    int C = input.size(1);
    int L = input.size(2);

    TORCH_CHECK(running_mean.is_cuda() && running_mean.is_contiguous() && running_mean.dtype() == torch::kFloat32,
                "running_mean must be CUDA contiguous float32 tensor");
    TORCH_CHECK(running_var.is_cuda() && running_var.is_contiguous() && running_var.dtype() == torch::kFloat32,
                "running_var must be CUDA contiguous float32 tensor");
    TORCH_CHECK(running_mean.numel() == C && running_var.numel() == C,
                "running_mean and running_var must have length C");

    // Prepare weight and bias pointers
    const float* weight_ptr = nullptr;
    const float* bias_ptr = nullptr;
    if (weight.has_value() && weight->defined()) {
        TORCH_CHECK(weight->is_cuda() && weight->is_contiguous() && weight->dtype() == torch::kFloat32,
                    "weight must be CUDA contiguous float32 tensor");
        TORCH_CHECK(weight->numel() == C, "weight must have length C");
        weight_ptr = weight->data_ptr<float>();
    }
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->is_cuda() && bias->is_contiguous() && bias->dtype() == torch::kFloat32,
                    "bias must be CUDA contiguous float32 tensor");
        TORCH_CHECK(bias->numel() == C, "bias must have length C");
        bias_ptr = bias->data_ptr<float>();
    }

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    op_batch_norm_1d_training_launcher(input.data_ptr<float>(),
                                       output.data_ptr<float>(),
                                       weight_ptr,
                                       bias_ptr,
                                       running_mean.data_ptr<float>(),
                                       running_var.data_ptr<float>(),
                                       N, C, L,
                                       static_cast<float>(eps),
                                       static_cast<float>(momentum),
                                       stream);

    return output;
}

// Registration function
void register_custom_ops(pybind11::module& m) {
    m.def("op_unfold", &op_unfold_forward,
          "Unfold operation (stride=1, padding=0)",
          py::arg("x"),
          py::arg("kernel_h"),
          py::arg("kernel_w"),
          py::arg("dilation_h"),
          py::arg("dilation_w"));

    m.def("op_batch_norm_1d_training", &op_batch_norm_1d_training_forward,
          "BatchNorm1d training mode",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"),
          py::arg("momentum"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
