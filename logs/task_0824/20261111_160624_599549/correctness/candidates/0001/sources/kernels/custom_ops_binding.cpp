// custom_ops_binding.cpp
// Tensor wrappers and pybind registrations for custom_ops.cu launchers.
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include "../binding_registry.h"

// Declare extern "C" launchers from custom_ops.cu
extern "C" void batch_norm_2d_forward_launcher(
    const float* input, float* output,
    const float* weight, const float* bias,
    const float* running_mean, const float* running_var,
    float eps, bool training,
    int64_t num_elements, int64_t C, int64_t HW,
    cudaStream_t stream);

extern "C" void permute_contiguous_4d_launcher(
    const float* input, float* output,
    int64_t in_shape[4], int64_t perm[4],
    int64_t num_elements, cudaStream_t stream);

extern "C" void view_contiguous_copy_launcher(
    const float* input, float* output,
    int64_t num_elements, cudaStream_t stream);

extern "C" void max_pool1d_forward_launcher(
    const float* input, float* output,
    int64_t B, int64_t H, int64_t L, int64_t L_out,
    int kernel_size, int stride, int padding, int dilation,
    bool ceil_mode, cudaStream_t stream);

// -----------------------------------------------------------------------------
// Helper: validate CUDA float32 contiguous tensor
static void check_cuda_float32_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

// -----------------------------------------------------------------------------
// 1. batch_norm_2d_forward
torch::Tensor batch_norm_2d_forward(
    torch::Tensor input,
    c10::optional<torch::Tensor> weight,
    c10::optional<torch::Tensor> bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    double eps,
    bool training) {

    check_cuda_float32_contiguous(input, "input");
    check_cuda_float32_contiguous(running_mean, "running_mean");
    check_cuda_float32_contiguous(running_var, "running_var");

    TORCH_CHECK(input.dim() == 4, "input must be 4D (B,C,H,W)");
    int64_t C = input.size(1);
    TORCH_CHECK(running_mean.numel() == C, "running_mean size must match C");
    TORCH_CHECK(running_var.numel() == C, "running_var size must match C");

    torch::Tensor weight_tensor;
    if (weight.has_value()) {
        weight_tensor = weight.value();
        check_cuda_float32_contiguous(weight_tensor, "weight");
        TORCH_CHECK(weight_tensor.numel() == C, "weight size must match C");
    }
    torch::Tensor bias_tensor;
    if (bias.has_value()) {
        bias_tensor = bias.value();
        check_cuda_float32_contiguous(bias_tensor, "bias");
        TORCH_CHECK(bias_tensor.numel() == C, "bias size must match C");
    }

    TORCH_CHECK(!training, "training=true is not supported for BatchNorm2d in this extension");

    auto output = torch::empty_like(input);
    int64_t num_elements = input.numel();
    int64_t HW = input.size(2) * input.size(3);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    batch_norm_2d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        weight_tensor.defined() ? weight_tensor.data_ptr<float>() : nullptr,
        bias_tensor.defined() ? bias_tensor.data_ptr<float>() : nullptr,
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        static_cast<float>(eps),
        false,
        num_elements,
        C,
        HW,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// 2. permute_contiguous_4d
torch::Tensor permute_contiguous_4d(
    torch::Tensor input,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) {

    check_cuda_float32_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");

    int64_t perm[4] = {dim0, dim1, dim2, dim3};
    // Validate permutation
    bool seen[4] = {false, false, false, false};
    for (int i = 0; i < 4; ++i) {
        TORCH_CHECK(perm[i] >= 0 && perm[i] < 4, "permutation dims must be in [0,3]");
        TORCH_CHECK(!seen[perm[i]], "duplicate perm dim");
        seen[perm[i]] = true;
    }

    std::vector<int64_t> output_shape(4);
    for (int i = 0; i < 4; ++i) {
        output_shape[i] = input.size(perm[i]);
    }
    auto output = torch::empty(output_shape, input.options());

    int64_t in_shape[4];
    for (int i = 0; i < 4; ++i) in_shape[i] = input.size(i);
    int64_t num_elements = input.numel();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    permute_contiguous_4d_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        in_shape,
        perm,
        num_elements,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// 3. view_4d (reshape from 4D contiguous to 3D)
torch::Tensor view_4d(
    torch::Tensor input,
    int64_t size0, int64_t size1, int64_t size2) {

    check_cuda_float32_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 4, "input must be 4D");

    int64_t total = input.numel();
    if (size2 == -1) {
        TORCH_CHECK(size0 > 0 && size1 > 0, "size0 and size1 must be positive when size2 is -1");
        size2 = total / (size0 * size1);
    }
    TORCH_CHECK(size0 * size1 * size2 == total, "view sizes do not match total elements");

    // Since view is just a reshape of contiguous data, we can return a view (no copy needed).
    // But to be safe and match the contract (returns a new tensor), we create a view using torch::from_blob? 
    // Better: return input.view({size0, size1, size2});  This is allowed? torch::view is not a torch operator? It's a Tensor method, not torch:: namespace. We can use Tensor.view.
    // The instruction says "不得调用 torch:: 算子完成计算", but view is just metadata operation. We can use Tensor.view.
    return input.view({size0, size1, size2});
}

// -----------------------------------------------------------------------------
// 4. max_pool1d_forward
torch::Tensor max_pool1d_forward(
    torch::Tensor input,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, bool ceil_mode) {

    check_cuda_float32_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B,H,L)");
    TORCH_CHECK(kernel_size > 0 && stride > 0 && padding >= 0 && dilation > 0,
                "invalid max_pool parameters");

    int64_t B = input.size(0);
    int64_t H = input.size(1);
    int64_t L = input.size(2);

    int64_t L_out;
    if (ceil_mode) {
        L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    } else {
        L_out = (L + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        // floor mode: integer division already floors positive values
    }
    // Ensure non-negative
    TORCH_CHECK(L_out > 0, "L_out must be positive");

    auto output = torch::empty({B, H, L_out}, input.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    max_pool1d_forward_launcher(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        B, H, L, L_out,
        static_cast<int>(kernel_size),
        static_cast<int>(stride),
        static_cast<int>(padding),
        static_cast<int>(dilation),
        ceil_mode,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// 5. view_5d (reshape from 3D contiguous to 4D)
torch::Tensor view_5d(
    torch::Tensor input,
    int64_t size0, int64_t size1, int64_t size2, int64_t size3) {

    check_cuda_float32_contiguous(input, "input");
    TORCH_CHECK(input.dim() == 3, "input must be 3D");

    int64_t total = input.numel();
    if (size2 == -1) {
        TORCH_CHECK(size0 > 0 && size1 > 0 && size3 > 0,
                   "size0,size1,size3 must be positive when size2=-1");
        size2 = total / (size0 * size1 * size3);
    }
    TORCH_CHECK(size0 * size1 * size2 * size3 == total, "view sizes do not match total elements");

    return input.view({size0, size1, size2, size3});
}

// -----------------------------------------------------------------------------
// 6. permute_contiguous_4d_v2 (same as permute_contiguous_4d, but input may be 4D with last dim 1)
torch::Tensor permute_contiguous_4d_v2(
    torch::Tensor input,
    int64_t dim0, int64_t dim1, int64_t dim2, int64_t dim3) {

    // identical implementation to permute_contiguous_4d
    return permute_contiguous_4d(input, dim0, dim1, dim2, dim3);
}

// -----------------------------------------------------------------------------
// Register bindings
void register_custom_ops(pybind11::module& m) {
    m.def("batch_norm_2d_forward", &batch_norm_2d_forward,
          "BatchNorm2d forward (inference only)",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("running_mean"), py::arg("running_var"), py::arg("eps"),
          py::arg("training"));

    m.def("permute_contiguous_4d", &permute_contiguous_4d,
          "Permute and make contiguous",
          py::arg("input"), py::arg("dim0"), py::arg("dim1"), py::arg("dim2"), py::arg("dim3"));

    m.def("view_4d", &view_4d,
          "View 4D tensor as 3D",
          py::arg("input"), py::arg("size0"), py::arg("size1"), py::arg("size2"));

    m.def("max_pool1d_forward", &max_pool1d_forward,
          "MaxPool1d forward",
          py::arg("input"), py::arg("kernel_size"), py::arg("stride"),
          py::arg("padding"), py::arg("dilation"), py::arg("ceil_mode"));

    m.def("view_5d", &view_5d,
          "View 3D tensor as 4D",
          py::arg("input"), py::arg("size0"), py::arg("size1"), py::arg("size2"), py::arg("size3"));

    m.def("permute_contiguous_4d_v2", &permute_contiguous_4d_v2,
          "Permute and make contiguous (v2)",
          py::arg("input"), py::arg("dim0"), py::arg("dim1"), py::arg("dim2"), py::arg("dim3"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
