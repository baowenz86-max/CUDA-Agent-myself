#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void mha_forward_launcher(
    const float* x,
    const float* in_proj_weight,
    const float* in_proj_bias,
    const float* out_proj_weight,
    const float* out_proj_bias,
    float* attn_output,
    float* attn_weights_avg,
    int B, int L, int E, int H,
    cudaStream_t stream
);

std::tuple<torch::Tensor, torch::Tensor> mha_forward(
    torch::Tensor x,
    torch::Tensor in_proj_weight,
    torch::Tensor in_proj_bias,
    torch::Tensor out_proj_weight,
    torch::Tensor out_proj_bias)
{
    // Validate device and dtype
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(in_proj_weight.is_cuda(), "in_proj_weight must be a CUDA tensor");
    TORCH_CHECK(in_proj_bias.is_cuda(), "in_proj_bias must be a CUDA tensor");
    TORCH_CHECK(out_proj_weight.is_cuda(), "out_proj_weight must be a CUDA tensor");
    TORCH_CHECK(out_proj_bias.is_cuda(), "out_proj_bias must be a CUDA tensor");

    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(in_proj_weight.dtype() == torch::kFloat32, "in_proj_weight must be float32");
    TORCH_CHECK(in_proj_bias.dtype() == torch::kFloat32, "in_proj_bias must be float32");
    TORCH_CHECK(out_proj_weight.dtype() == torch::kFloat32, "out_proj_weight must be float32");
    TORCH_CHECK(out_proj_bias.dtype() == torch::kFloat32, "out_proj_bias must be float32");

    // Make x contiguous if not already; weights per contract are contiguous but ensure for safety
    x = x.contiguous();
    in_proj_weight = in_proj_weight.contiguous();
    in_proj_bias = in_proj_bias.contiguous();
    out_proj_weight = out_proj_weight.contiguous();
    out_proj_bias = out_proj_bias.contiguous();

    // Parse dimensions from input shape
    int B = x.size(0);
    int L = x.size(1);
    int E = x.size(2);
    int H = 4; // num_heads fixed from model init

    TORCH_CHECK(in_proj_weight.size(0) == 3 * E, "in_proj_weight must be [3E, E]");
    TORCH_CHECK(in_proj_weight.size(1) == E, "in_proj_weight must be [3E, E]");
    TORCH_CHECK(in_proj_bias.size(0) == 3 * E, "in_proj_bias length must be 3E");
    TORCH_CHECK(out_proj_weight.size(0) == E && out_proj_weight.size(1) == E, "out_proj_weight must be [E, E]");
    TORCH_CHECK(out_proj_bias.size(0) == E, "out_proj_bias length must be E");

    // Allocate output tensors
    auto attn_output = torch::empty_like(x); // [B, L, E]
    auto attn_weights_avg = torch::empty({L, B, B}, x.options()); // [L, B, B]

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    mha_forward_launcher(
        x.data_ptr<float>(),
        in_proj_weight.data_ptr<float>(),
        in_proj_bias.data_ptr<float>(),
        out_proj_weight.data_ptr<float>(),
        out_proj_bias.data_ptr<float>(),
        attn_output.data_ptr<float>(),
        attn_weights_avg.data_ptr<float>(),
        B, L, E, H,
        stream
    );

    // Check for CUDA errors (optional but recommended)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA error in mha_forward: ") + cudaGetErrorString(err));
    }

    return std::make_tuple(attn_output, attn_weights_avg);
}

void register_mha_forward(pybind11::module& m) {
    m.def("mha_forward", &mha_forward,
          "MultiheadAttention forward",
          py::arg("x"),
          py::arg("in_proj_weight"),
          py::arg("in_proj_bias"),
          py::arg("out_proj_weight"),
          py::arg("out_proj_bias"));
}

REGISTER_BINDING(mha_forward, register_mha_forward);
