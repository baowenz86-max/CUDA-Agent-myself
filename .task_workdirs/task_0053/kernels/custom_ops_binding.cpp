#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void log_softmax_forward_launcher(const float* input, float* output,
                                             int B, int C, int D,
                                             long long stride_b, long long stride_c, long long stride_d,
                                             long long out_stride_b, long long out_stride_c, long long out_stride_d,
                                             cudaStream_t stream);

extern "C" void softmax_forward_launcher(const float* input, float* output,
                                         int B, int C, int D,
                                         long long stride_b, long long stride_c, long long stride_d,
                                         long long out_stride_b, long long out_stride_c, long long out_stride_d,
                                         cudaStream_t stream);

extern "C" void kl_div_loss_forward_launcher(const float* input_log, const float* target_prob,
                                             float* output, int B, int C, int D,
                                             long long stride_b, long long stride_c, long long stride_d,
                                             cudaStream_t stream);

extern "C" void cross_entropy_loss_forward_launcher(const float* input, const int64_t* target,
                                                    float* output, int B, int C, int D,
                                                    long long stride_b, long long stride_c, long long stride_d,
                                                    long long target_stride_b, long long target_stride_d,
                                                    cudaStream_t stream);

extern "C" void add_forward_launcher(const float* a, const float* b, float* output, cudaStream_t stream);

extern "C" void remainder_forward_launcher(const float* input, float divisor, float* output, cudaStream_t stream);

// Helper to get current CUDA stream
static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// -----------------------------------------------------------------------------
// log_softmax_forward
// -----------------------------------------------------------------------------
torch::Tensor log_softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, C, D)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);

    auto output = torch::empty_like(input);

    // Strides for input (may be non-contiguous)
    long long stride_b = input.stride(0);
    long long stride_c = input.stride(1);
    long long stride_d = input.stride(2);

    // Output is contiguous (torch::empty_like of contiguous? Actually empty_like preserves strides, so use contiguous output)
    // Force contiguous output for simplicity, matching launcher assumption of contiguous output.
    auto output_contig = output.contiguous();
    long long out_stride_b = output_contig.stride(0);
    long long out_stride_c = output_contig.stride(1);
    long long out_stride_d = output_contig.stride(2);

    cudaStream_t stream = getCurrentStream();

    log_softmax_forward_launcher(
        input.data_ptr<float>(),
        output_contig.data_ptr<float>(),
        B, C, D,
        stride_b, stride_c, stride_d,
        out_stride_b, out_stride_c, out_stride_d,
        stream
    );

    return output_contig;
}

// -----------------------------------------------------------------------------
// softmax_forward
// -----------------------------------------------------------------------------
torch::Tensor softmax_forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.dim() == 3, "Input must be 3D (B, C, D)");
    TORCH_CHECK(dim == 1, "Only dim=1 is supported");

    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);

    auto output = torch::empty_like(input);

    long long stride_b = input.stride(0);
    long long stride_c = input.stride(1);
    long long stride_d = input.stride(2);

    auto output_contig = output.contiguous();
    long long out_stride_b = output_contig.stride(0);
    long long out_stride_c = output_contig.stride(1);
    long long out_stride_d = output_contig.stride(2);

    cudaStream_t stream = getCurrentStream();

    softmax_forward_launcher(
        input.data_ptr<float>(),
        output_contig.data_ptr<float>(),
        B, C, D,
        stride_b, stride_c, stride_d,
        out_stride_b, out_stride_c, out_stride_d,
        stream
    );

    return output_contig;
}

// -----------------------------------------------------------------------------
// kl_div_loss_forward
// -----------------------------------------------------------------------------
torch::Tensor kl_div_loss_forward(torch::Tensor input_log, torch::Tensor target_prob) {
    TORCH_CHECK(input_log.is_cuda(), "input_log must be CUDA tensor");
    TORCH_CHECK(target_prob.is_cuda(), "target_prob must be CUDA tensor");
    TORCH_CHECK(input_log.dtype() == torch::kFloat32, "input_log must be float32");
    TORCH_CHECK(target_prob.dtype() == torch::kFloat32, "target_prob must be float32");
    TORCH_CHECK(input_log.dim() == 3, "input_log must be 3D");
    TORCH_CHECK(target_prob.dim() == 3, "target_prob must be 3D");
    TORCH_CHECK(input_log.size(0) == target_prob.size(0) && input_log.size(1) == target_prob.size(1) && input_log.size(2) == target_prob.size(2),
                "Shapes must match");

    int B = input_log.size(0);
    int C = input_log.size(1);
    int D = input_log.size(2);

    // Allocate scalar output tensor
    auto output = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(input_log.device()));

    long long stride_b = input_log.stride(0);
    long long stride_c = input_log.stride(1);
    long long stride_d = input_log.stride(2);
    // Note: target_prob should have same shape and strides? Not necessarily, but launcher expects same strides as provided for input_log.
    // We need to ensure both tensors have same shape, which is checked. Strides can differ but launcher ignores target strides and assumes same as input_log?
    // Looking at kl_div_loss_kernel, it uses stride_b, stride_c, stride_d for both input_log and target_prob offset calculation,
    // thus it assumes same strides for both. To be safe, we make both contiguous and use contiguous strides for both.
    auto input_log_contig = input_log.contiguous();
    auto target_prob_contig = target_prob.contiguous();
    stride_b = input_log_contig.stride(0);
    stride_c = input_log_contig.stride(1);
    stride_d = input_log_contig.stride(2);

    cudaStream_t stream = getCurrentStream();

    kl_div_loss_forward_launcher(
        input_log_contig.data_ptr<float>(),
        target_prob_contig.data_ptr<float>(),
        output.data_ptr<float>(),
        B, C, D,
        stride_b, stride_c, stride_d,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// cross_entropy_loss_forward
// -----------------------------------------------------------------------------
torch::Tensor cross_entropy_loss_forward(torch::Tensor input, torch::Tensor target) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(target.is_cuda(), "target must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(target.dtype() == torch::kInt64, "target must be int64");
    TORCH_CHECK(input.dim() == 3, "input must be 3D (B, C, D)");
    TORCH_CHECK(target.dim() == 2, "target must be 2D (B, D)");
    TORCH_CHECK(input.size(0) == target.size(0) && input.size(2) == target.size(1),
                "Batch and D dimensions must match");

    int B = input.size(0);
    int C = input.size(1);
    int D = input.size(2);

    auto output = torch::zeros({}, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));

    // Ensure input is contiguous for kernel simplicity (kernel uses stride_b, stride_c, stride_d)
    auto input_contig = input.contiguous();
    auto target_contig = target.contiguous();

    long long stride_b = input_contig.stride(0);
    long long stride_c = input_contig.stride(1);
    long long stride_d = input_contig.stride(2);
    long long target_stride_b = target_contig.stride(0);
    long long target_stride_d = target_contig.stride(1);

    cudaStream_t stream = getCurrentStream();

    cross_entropy_loss_forward_launcher(
        input_contig.data_ptr<float>(),
        target_contig.data_ptr<int64_t>(),
        output.data_ptr<float>(),
        B, C, D,
        stride_b, stride_c, stride_d,
        target_stride_b, target_stride_d,
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// add_forward
// -----------------------------------------------------------------------------
torch::Tensor add_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Both inputs must be CUDA tensors");
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32, "Both inputs must be float32");
    TORCH_CHECK(a.numel() == 1 && b.numel() == 1, "Both inputs must be scalar tensors (shape [])");

    auto output = torch::empty_like(a);

    cudaStream_t stream = getCurrentStream();

    add_forward_launcher(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        output.data_ptr<float>(),
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// remainder_forward
// -----------------------------------------------------------------------------
torch::Tensor remainder_forward(torch::Tensor input, double divisor) {
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input must be float32");
    TORCH_CHECK(input.numel() == 1, "Input must be scalar tensor (shape [])");

    auto output = torch::empty_like(input);

    cudaStream_t stream = getCurrentStream();

    remainder_forward_launcher(
        input.data_ptr<float>(),
        static_cast<float>(divisor),
        output.data_ptr<float>(),
        stream
    );

    return output;
}

// -----------------------------------------------------------------------------
// Registration
// -----------------------------------------------------------------------------
void register_custom_ops(pybind11::module& m) {
    m.def("log_softmax_forward", &log_softmax_forward, "LogSoftmax forward",
          py::arg("input"), py::arg("dim"));
    m.def("softmax_forward", &softmax_forward, "Softmax forward",
          py::arg("input"), py::arg("dim"));
    m.def("kl_div_loss_forward", &kl_div_loss_forward, "KLDivLoss forward",
          py::arg("input_log"), py::arg("target_prob"));
    m.def("cross_entropy_loss_forward", &cross_entropy_loss_forward, "CrossEntropyLoss forward",
          py::arg("input"), py::arg("target"));
    m.def("add_forward", &add_forward, "Add two scalar tensors",
          py::arg("a"), py::arg("b"));
    m.def("remainder_forward", &remainder_forward, "Remainder of scalar tensor by float",
          py::arg("input"), py::arg("divisor"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
