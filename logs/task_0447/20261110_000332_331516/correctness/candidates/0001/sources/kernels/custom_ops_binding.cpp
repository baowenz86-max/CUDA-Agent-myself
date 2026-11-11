#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
#include <vector>

// Declare launchers from custom_ops.cu
extern "C" void constant_pad3d_launcher(
    const float* input, float* output,
    int64_t N, int64_t C, int64_t D_in, int64_t H_in, int64_t W_in,
    int64_t D_out, int64_t H_out, int64_t W_out,
    int64_t pad_left, int64_t pad_right,
    int64_t pad_top, int64_t pad_bottom,
    int64_t pad_front, int64_t pad_back,
    cudaStream_t stream);

extern "C" void bias_add_launcher(
    const float* input, const float* bias, float* output,
    int64_t total_elements, int64_t C, int64_t spatial_size,
    cudaStream_t stream);

// PyTorch wrapper for constant_pad3d
// input: 5D tensor (N,C,D,H,W) on CUDA, float32, arbitrary stride allowed
// pad: tuple/list of 6 ints (left, right, top, bottom, front, back)
torch::Tensor constant_pad3d(torch::Tensor input, std::vector<int64_t> pad) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(pad.size() == 6, "pad must have 6 elements");

    auto N = input.size(0);
    auto C = input.size(1);
    auto D_in = input.size(2);
    auto H_in = input.size(3);
    auto W_in = input.size(4);

    int64_t pad_left = pad[0];
    int64_t pad_right = pad[1];
    int64_t pad_top = pad[2];
    int64_t pad_bottom = pad[3];
    int64_t pad_front = pad[4];
    int64_t pad_back = pad[5];

    int64_t D_out = D_in + pad_front + pad_back;
    int64_t H_out = H_in + pad_top + pad_bottom;
    int64_t W_out = W_in + pad_left + pad_right;

    // Create output tensor with same dtype and device, contiguous
    auto output = torch::empty({N, C, D_out, H_out, W_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    constant_pad3d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, D_in, H_in, W_in,
        D_out, H_out, W_out,
        pad_left, pad_right,
        pad_top, pad_bottom,
        pad_front, pad_back,
        stream
    );

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "constant_pad3d kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// PyTorch wrapper for bias_add
// input: 5D tensor (N,C,D,H,W) on CUDA, float32, arbitrary stride allowed
// bias: tensor of shape [C] or [C,1,1,1] on CUDA, float32
torch::Tensor bias_add(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(input.dim() == 5, "input must be 5D");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "bias must be float32");

    auto C = input.size(1);
    TORCH_CHECK(bias.numel() == C, "bias must have C elements");

    // Ensure bias is contiguous and flatten to 1D pointer
    auto bias_contig = bias.contiguous();

    auto N = input.size(0);
    auto D = input.size(2);
    auto H = input.size(3);
    auto W = input.size(4);
    int64_t spatial_size = D * H * W;
    int64_t total_elements = input.numel();

    // Create output tensor with same shape as input, contiguous
    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bias_add_launcher(
        input.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        total_elements,
        C,
        spatial_size,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bias_add kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("constant_pad3d", &constant_pad3d, "ConstantPad3d forward",
          py::arg("input"), py::arg("pad"));
    m.def("bias_add", &bias_add, "Bias add forward",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
