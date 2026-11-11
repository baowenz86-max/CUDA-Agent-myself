#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declarations of launchers from custom_ops.cu
extern "C" void custom_mean_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int in_stride_N, int in_stride_C, int in_stride_H, int in_stride_W,
    int out_stride_N, int out_stride_C, int out_stride_W,
    cudaStream_t stream
);

extern "C" void custom_gt_launcher(
    const float* input,
    const float* other,
    bool* output,
    int N, int C, int H, int W,
    int in_stride_N, int in_stride_C, int in_stride_H, int in_stride_W,
    int other_stride_N, int other_stride_C, int other_stride_W,
    int out_stride_N, int out_stride_C, int out_stride_H, int out_stride_W,
    cudaStream_t stream
);

// -----------------------------------------------------------------------------
// custom_mean forward wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_mean_forward(torch::Tensor input, int64_t dim, bool keepdim) {
    TORCH_CHECK(input.is_cuda(), "custom_mean: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_mean: input must be float32");
    TORCH_CHECK(dim >= 0 && dim < input.dim(), "custom_mean: dim out of range");

    // For this task, only dim=2 and keepdim=true are used; still check broadly.
    TORCH_CHECK(keepdim, "custom_mean: keepdim=false is not supported by the current launcher");
    TORCH_CHECK(input.dim() == 4, "custom_mean: only 4D tensors are supported");

    auto sizes = input.sizes();
    int64_t N = sizes[0];
    int64_t C = sizes[1];
    int64_t H = sizes[2];
    int64_t W = sizes[3];

    TORCH_CHECK(dim == 2, "custom_mean: only dim=2 is implemented");

    auto output = torch::empty({N, C, 1, W}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_mean_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(input.stride(0)),
        static_cast<int>(input.stride(1)),
        static_cast<int>(input.stride(2)),
        static_cast<int>(input.stride(3)),
        static_cast<int>(output.stride(0)),
        static_cast<int>(output.stride(1)),
        static_cast<int>(output.stride(3)),   // FIX: output w-stride is stride(3), not stride(2)
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_mean launcher failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// custom_gt forward wrapper
// -----------------------------------------------------------------------------
torch::Tensor custom_gt_forward(torch::Tensor input, torch::Tensor other) {
    TORCH_CHECK(input.is_cuda(), "custom_gt: input must be a CUDA tensor");
    TORCH_CHECK(other.is_cuda(), "custom_gt: other must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "custom_gt: input must be float32");
    TORCH_CHECK(other.dtype() == torch::kFloat32, "custom_gt: other must be float32");
    TORCH_CHECK(input.dim() == 4, "custom_gt: input must be 4D");
    TORCH_CHECK(other.dim() == 4, "custom_gt: other must be 4D");

    // Check broadcast-compatible with specific shapes: input [N,C,H,W], other [N,C,1,W]
    TORCH_CHECK(input.size(0) == other.size(0) && input.size(1) == other.size(1),
                "custom_gt: N and C must match");
    TORCH_CHECK(other.size(2) == 1, "custom_gt: other's dim=2 must be 1");
    TORCH_CHECK(input.size(3) == other.size(3), "custom_gt: W must match");

    int64_t N = input.size(0);
    int64_t C = input.size(1);
    int64_t H = input.size(2);
    int64_t W = input.size(3);

    auto output = torch::empty({N, C, H, W}, input.options().dtype(torch::kBool));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_gt_launcher(
        input.data_ptr<float>(),
        other.data_ptr<float>(),
        output.data_ptr<bool>(),
        static_cast<int>(N),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(input.stride(0)),
        static_cast<int>(input.stride(1)),
        static_cast<int>(input.stride(2)),
        static_cast<int>(input.stride(3)),
        static_cast<int>(other.stride(0)),
        static_cast<int>(other.stride(1)),
        static_cast<int>(other.stride(3)),   // FIX: other w-stride is stride(3), not stride(2)
        static_cast<int>(output.stride(0)),
        static_cast<int>(output.stride(1)),
        static_cast<int>(output.stride(2)),
        static_cast<int>(output.stride(3)),
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "custom_gt launcher failed: ", cudaGetErrorString(err));

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops_binding(pybind11::module& m) {
    m.def("custom_mean", &custom_mean_forward,
          "Custom mean along dim (only dim=2, keepdim=True)",
          py::arg("input"), py::arg("dim"), py::arg("keepdim"));
    m.def("custom_gt", &custom_gt_forward,
          "Custom greater-than with broadcasting",
          py::arg("input"), py::arg("other"));
}

REGISTER_BINDING(custom_ops, register_custom_ops_binding);
