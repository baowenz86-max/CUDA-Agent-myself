#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void bilinear_upsample_launcher(
    const float* input,
    float* output,
    int N, int C, int H, int W,
    int H_out, int W_out,
    float scale_factor,
    cudaStream_t stream);

extern "C" void squeeze_identity_launcher(
    const float* input,
    float* output,
    cudaStream_t stream);

// Bilinear upsample wrapper
torch::Tensor bilinear_upsample(
    torch::Tensor x,
    double scale_factor,
    bool align_corners) {
    // Input validation
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N,C,H,W)");
    TORCH_CHECK(x.is_contiguous(), "Input must be contiguous");

    // align_corners must be false for this implementation
    TORCH_CHECK(!align_corners, "Only align_corners=False is supported");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);

    // Calculate output dimensions (integer multiplication, scale_factor is float)
    int H_out = static_cast<int>(H * scale_factor);
    int W_out = static_cast<int>(W * scale_factor);

    TORCH_CHECK(H_out > 0 && W_out > 0, "Invalid output dimensions");

    // Allocate output tensor
    auto output = torch::empty({N, C, H_out, W_out}, x.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Call launcher
    bilinear_upsample_launcher(
        x.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        H_out, W_out,
        static_cast<float>(scale_factor),
        stream);

    return output;
}

// Squeeze identity wrapper: for 4D tensor with no dims of size 1, torch.squeeze returns the same tensor
// We return the input tensor directly; no copy needed.
torch::Tensor squeeze_identity(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D for this contract");

    // Check if any dimension is 1; if so, PyTorch's squeeze would remove them, but our contract says reference has no dims of size 1.
    // To be safe, if any dim is 1, we return a view with squeezed dims (though this case shouldn't occur in verification).
    bool has_unit_dim = false;
    for (int i = 0; i < 4; ++i) {
        if (x.size(i) == 1) {
            has_unit_dim = true;
            break;
        }
    }

    if (has_unit_dim) {
        // Not expected in task, but implement proper behavior: return a view with squeezed dims
        // We'll use torch::squeeze (allowed for tensor manipulation, not computation)
        // However, we must avoid using torch:: ops for computation, but view operations are okay.
        // To keep this simple and aligned with task, we can use x.squeeze() which returns a view.
        // But note: this uses torch::squeeze which is a view operation, not a computation.
        // It's acceptable as per the restrictions.
        return x.squeeze();
    } else {
        // No unit dims, return the same tensor (as reference).
        return x;
    }
}

// Register both functions
void register_custom_ops(pybind11::module& m) {
    m.def("bilinear_upsample", &bilinear_upsample,
          "Bilinear upsample (scale_factor, align_corners=False)",
          pybind11::arg("x"),
          pybind11::arg("scale_factor"),
          pybind11::arg("align_corners"));

    m.def("squeeze_identity", &squeeze_identity,
          "Identity squeeze for 4D tensor without unit dims",
          pybind11::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
