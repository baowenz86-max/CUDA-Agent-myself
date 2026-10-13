#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void batch_norm_inference_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    const float* running_mean,
    const float* running_var,
    float eps,
    int N, int C, int H, int W,
    bool channels_last,
    cudaStream_t stream
);

extern "C" void group_norm_launcher(
    float* output,
    const float* input,
    const float* weight,
    const float* bias,
    float eps,
    int N, int C, int H, int W,
    int num_groups,
    bool channels_last,
    cudaStream_t stream
);

extern "C" void histc_launcher(
    float* output,
    const float* input,
    int num_elements,
    int bins,
    float min_val,
    float max_val,
    cudaStream_t stream
);

// Helper: check CUDA error after kernel launch
static void check_cuda_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TORCH_CHECK(false, msg, ": ", cudaGetErrorString(err));
    }
}

// batch_norm_inference wrapper
torch::Tensor batch_norm_inference(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps
) {
    // Input validation
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kFloat32, "weight must be CUDA float32");
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32, "bias must be CUDA float32");
    TORCH_CHECK(running_mean.is_cuda() && running_mean.dtype() == torch::kFloat32, "running_mean must be CUDA float32");
    TORCH_CHECK(running_var.is_cuda() && running_var.dtype() == torch::kFloat32, "running_var must be CUDA float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    TORCH_CHECK(weight.numel() == C && bias.numel() == C && running_mean.numel() == C && running_var.numel() == C,
        "weight/bias/running_mean/running_var size must equal C");

    // Detect channels-last format
    bool channels_last = false;
    if (input.is_contiguous(torch::MemoryFormat::ChannelsLast)) {
        channels_last = true;
    } else {
        input = input.contiguous();
    }

    auto output = torch::empty_like(input, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_inference_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        N, C, H, W,
        channels_last,
        stream
    );
    check_cuda_error("batch_norm_inference_launcher");

    return output;
}

// group_norm wrapper
torch::Tensor group_norm(
    torch::Tensor input,
    int64_t num_groups,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be 4D (N, C, H, W)");

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);

    TORCH_CHECK(num_groups > 0 && C % num_groups == 0, "invalid num_groups");

    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda() && weight.dtype() == torch::kFloat32, "weight must be CUDA float32 if provided");
        TORCH_CHECK(weight.numel() == C, "weight size must equal C");
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kFloat32, "bias must be CUDA float32 if provided");
        TORCH_CHECK(bias.numel() == C, "bias size must equal C");
    }

    input = input.contiguous();

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    group_norm_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        static_cast<float>(eps),
        N, C, H, W,
        static_cast<int>(num_groups),
        false,
        stream
    );
    check_cuda_error("group_norm_launcher");

    return output;
}

// histc wrapper
torch::Tensor histc(
    torch::Tensor input,
    int64_t bins,
    double min,
    double max
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bins > 0, "bins must be positive");

    input = input.contiguous();

    auto output = torch::empty({bins}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    histc_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        input.numel(),
        static_cast<int>(bins),
        static_cast<float>(min),
        static_cast<float>(max),
        stream
    );
    check_cuda_error("histc_launcher");

    return output;
}

// Register all operators
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_inference", &batch_norm_inference,
          "BatchNorm2d inference",
          py::arg("input"),
          py::arg("weight"),
          py::arg("bias"),
          py::arg("running_mean"),
          py::arg("running_var"),
          py::arg("eps"));

    // 显式设置默认值为 py::none()，确保 docstring 显示为 None
    m.def("group_norm", &group_norm,
          "GroupNorm",
          py::arg("input"),
          py::arg("num_groups"),
          py::arg("weight") = py::none(),
          py::arg("bias") = py::none(),
          py::arg("eps") = 1e-05);

    m.def("histc", &histc,
          "Histogram",
          py::arg("input"),
          py::arg("bins"),
          py::arg("min"),
          py::arg("max"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
