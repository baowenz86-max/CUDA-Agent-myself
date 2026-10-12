// Binding for add_pos_enc: out = x + pos_enc[:, :seq_len, :]
// Only float32, contiguous, CUDA tensors supported.

#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include "../binding_registry.h"

namespace py = pybind11;

// Declaration of launcher from add_pos_enc_kernel.cu
extern "C" void add_pos_enc_launcher(
    float* output,
    const float* input,
    const float* pos_enc,
    int batch_size,
    int seq_len,
    int embed_dim,
    int pos_dim,
    cudaStream_t stream
);

torch::Tensor add_pos_enc(torch::Tensor input, torch::Tensor pos_enc) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(pos_enc.is_cuda(), "pos_enc must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(pos_enc.dtype() == torch::kFloat32, "pos_enc must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(pos_enc.is_contiguous(), "pos_enc must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (batch, seq_len, embed_dim)");
    TORCH_CHECK(pos_enc.dim() == 3, "pos_enc must be 3D (1, pos_dim, embed_dim)");
    TORCH_CHECK(pos_enc.size(0) == 1, "pos_enc batch dimension must be 1");
    TORCH_CHECK(pos_enc.size(2) == input.size(2), "embed_dim mismatch");
    TORCH_CHECK(input.size(1) <= pos_enc.size(1), "seq_len exceeds pos_dim");

    int batch_size = input.size(0);
    int seq_len = input.size(1);
    int embed_dim = input.size(2);
    int pos_dim = pos_enc.size(1);

    auto output = torch::empty_like(input);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    add_pos_enc_launcher(
        output.data_ptr<float>(),
        input.data_ptr<float>(),
        pos_enc.data_ptr<float>(),
        batch_size, seq_len, embed_dim, pos_dim,
        stream
    );

    return output;
}

void register_add_pos_enc(py::module& m) {
    m.def("add_pos_enc", &add_pos_enc, "Add positional encoding (out = x + pos_enc)",
          py::arg("input"), py::arg("pos_enc"));
}

REGISTER_BINDING(add_pos_enc, register_add_pos_enc);
