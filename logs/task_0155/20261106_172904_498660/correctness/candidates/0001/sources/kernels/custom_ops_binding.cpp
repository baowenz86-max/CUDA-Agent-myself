// kernels/custom_ops_binding.cpp
// PyTorch bindings for conv3d (cuDNN) and tanh (custom CUDA kernel).
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>
#include <vector>
#include <stdexcept>
#include "../binding_registry.h"

// Declaration of launcher from custom_ops.cu
extern "C" void tanh_forward_launcher(const float* input,
                                      float* output,
                                      int64_t numel,
                                      cudaStream_t stream);

namespace {

// Helper to get cuDNN error string
const char* cudnnGetErrorStringSafe(cudnnStatus_t status) {
    switch (status) {
        case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        case CUDNN_STATUS_LICENSE_ERROR: return "CUDNN_STATUS_LICENSE_ERROR";
        default: return "UNKNOWN_CUDNN_STATUS";
    }
}

// Helper to convert tuple of ints to vector<int>
std::vector<int> tuple_to_int_vector(const pybind11::tuple& t) {
    std::vector<int> vec;
    for (auto item : t) {
        vec.push_back(item.cast<int>());
    }
    return vec;
}

// Helper to convert either int or tuple/list to 3-element vector for conv3d
std::vector<int> get_3d_param(const pybind11::object& obj, const char* param_name) {
    std::vector<int> result;
    if (pybind11::isinstance<pybind11::int_>(obj)) {
        int val = obj.cast<int>();
        result = {val, val, val};
    } else if (pybind11::isinstance<pybind11::tuple>(obj)) {
        auto t = obj.cast<pybind11::tuple>();
        if (t.size() != 3) {
            throw std::runtime_error(std::string(param_name) + " must be a 3-tuple or int");
        }
        result = tuple_to_int_vector(t);
    } else if (pybind11::isinstance<pybind11::list>(obj)) {
        auto l = obj.cast<pybind11::list>();
        if (l.size() != 3) {
            throw std::runtime_error(std::string(param_name) + " must be a 3-list or int");
        }
        result = tuple_to_int_vector(pybind11::reinterpret_borrow<pybind11::tuple>(l));
    } else {
        throw std::runtime_error(std::string(param_name) + " must be int, tuple, or list");
    }
    return result;
}

// RAII wrapper for cuDNN handle
typedef struct CudnnHandleWrapper {
    cudnnHandle_t handle;
    CudnnHandleWrapper() {
        cudnnStatus_t status = cudnnCreate(&handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuDNN handle: " + std::string(cudnnGetErrorStringSafe(status)));
        }
    }
    ~CudnnHandleWrapper() {
        cudnnDestroy(handle);
    }
    operator cudnnHandle_t() const { return handle; }
} CudnnHandleWrapper;

// RAII wrapper for tensor descriptor
typedef struct TensorDescriptorWrapper {
    cudnnTensorDescriptor_t desc;
    TensorDescriptorWrapper() : desc(nullptr) {}
    ~TensorDescriptorWrapper() {
        if (desc) cudnnDestroyTensorDescriptor(desc);
    }
    cudnnTensorDescriptor_t get() const { return desc; }
    void create() {
        cudnnStatus_t status = cudnnCreateTensorDescriptor(&desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create tensor descriptor: " + std::string(cudnnGetErrorStringSafe(status)));
        }
    }
} TensorDescriptorWrapper;

// RAII wrapper for filter descriptor
typedef struct FilterDescriptorWrapper {
    cudnnFilterDescriptor_t desc;
    FilterDescriptorWrapper() : desc(nullptr) {}
    ~FilterDescriptorWrapper() {
        if (desc) cudnnDestroyFilterDescriptor(desc);
    }
    cudnnFilterDescriptor_t get() const { return desc; }
    void create() {
        cudnnStatus_t status = cudnnCreateFilterDescriptor(&desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create filter descriptor: " + std::string(cudnnGetErrorStringSafe(status)));
        }
    }
} FilterDescriptorWrapper;

// RAII wrapper for convolution descriptor
typedef struct ConvolutionDescriptorWrapper {
    cudnnConvolutionDescriptor_t desc;
    ConvolutionDescriptorWrapper() : desc(nullptr) {}
    ~ConvolutionDescriptorWrapper() {
        if (desc) cudnnDestroyConvolutionDescriptor(desc);
    }
    cudnnConvolutionDescriptor_t get() const { return desc; }
    void create() {
        cudnnStatus_t status = cudnnCreateConvolutionDescriptor(&desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create convolution descriptor: " + std::string(cudnnGetErrorStringSafe(status)));
        }
    }
} ConvolutionDescriptorWrapper;

}  // namespace

// =============================================================================
// conv3d_forward binding
// =============================================================================
torch::Tensor conv3d_forward(torch::Tensor input,
                             torch::Tensor weight,
                             c10::optional<torch::Tensor> bias,
                             pybind11::object stride,
                             pybind11::object padding,
                             pybind11::object dilation) {
    // Validate input and weight
    TORCH_CHECK(input.is_cuda(), "conv3d_forward: input must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv3d_forward: weight must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "conv3d_forward: input must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "conv3d_forward: weight must be float32");
    TORCH_CHECK(input.dim() == 5, "conv3d_forward: input must be 5D (N, C, D, H, W)");
    TORCH_CHECK(weight.dim() == 5, "conv3d_forward: weight must be 5D (C_out, C_in, KD, KH, KW)");

    // Ensure tensors are contiguous (cuDNN requires contiguous)
    auto input_contig = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig = bias.has_value() ? bias->contiguous() : torch::Tensor();

    // Check bias if provided
    if (bias.has_value()) {
        TORCH_CHECK(bias_contig.is_cuda(), "conv3d_forward: bias must be a CUDA tensor");
        TORCH_CHECK(bias_contig.dtype() == torch::kFloat32, "conv3d_forward: bias must be float32");
        TORCH_CHECK(bias_contig.dim() == 1, "conv3d_forward: bias must be 1D");
        TORCH_CHECK(bias_contig.size(0) == weight_contig.size(0), "conv3d_forward: bias size must match output channels");
    }

    // Parse stride, padding, dilation (all must be 3-element after conversion)
    std::vector<int> stride_vec = get_3d_param(stride, "stride");
    std::vector<int> padding_vec = get_3d_param(padding, "padding");
    std::vector<int> dilation_vec = get_3d_param(dilation, "dilation");

    // Validate positive values
    for (int i = 0; i < 3; ++i) {
        TORCH_CHECK(stride_vec[i] >= 1, "conv3d_forward: stride must be >= 1");
        TORCH_CHECK(padding_vec[i] >= 0, "conv3d_forward: padding must be >= 0");
        TORCH_CHECK(dilation_vec[i] >= 1, "conv3d_forward: dilation must be >= 1");
    }

    // Dimensions
    const int64_t N = input_contig.size(0);
    const int64_t C_in = input_contig.size(1);
    const int64_t D_in = input_contig.size(2);
    const int64_t H_in = input_contig.size(3);
    const int64_t W_in = input_contig.size(4);
    const int64_t C_out = weight_contig.size(0);
    const int64_t C_in_weight = weight_contig.size(1);
    const int64_t KD = weight_contig.size(2);
    const int64_t KH = weight_contig.size(3);
    const int64_t KW = weight_contig.size(4);

    TORCH_CHECK(C_in == C_in_weight, "conv3d_forward: input channels mismatch");

    // Output spatial dimensions
    auto calc_out_dim = [](int64_t in_dim, int pad, int dilation_, int kernel, int stride_) {
        return (in_dim + 2 * pad - dilation_ * (kernel - 1) - 1) / stride_ + 1;
    };
    const int64_t D_out = calc_out_dim(D_in, padding_vec[0], dilation_vec[0], KD, stride_vec[0]);
    const int64_t H_out = calc_out_dim(H_in, padding_vec[1], dilation_vec[1], KH, stride_vec[1]);
    const int64_t W_out = calc_out_dim(W_in, padding_vec[2], dilation_vec[2], KW, stride_vec[2]);

    TORCH_CHECK(D_out > 0 && H_out > 0 && W_out > 0, "conv3d_forward: computed output dims must be positive");

    // Allocate output tensor
    auto output = torch::empty({N, C_out, D_out, H_out, W_out}, input_contig.options());

    // Get current CUDA stream
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // cuDNN handle (create per call for simplicity; could cache, but for performance maybe static later)
    CudnnHandleWrapper cudnn_handle;

    // Set stream for handle
    cudnnStatus_t status = cudnnSetStream(cudnn_handle, stream);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnSetStream failed: " + std::string(cudnnGetErrorStringSafe(status)));
    }

    // Create tensor descriptors
    TensorDescriptorWrapper input_desc;
    input_desc.create();
    status = cudnnSetTensorNdDescriptor(input_desc.get(), CUDNN_DATA_FLOAT, 5,
                                        std::vector<int64_t>{N, C_in, D_in, H_in, W_in}.data());
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnSetTensorNdDescriptor input failed: " + std::string(cudnnGetErrorStringSafe(status)));
    }
    // Set strides? cuDNN uses default NCHW format which matches contiguous PyTorch.
    // No explicit stride needed.

    TensorDescriptorWrapper weight_desc;
    weight_desc.create();
    status = cudnnSetFilterNdDescriptor(weight_desc.get(), CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 5,
                                        std::vector<int64_t>{C_out, C_in, KD, KH, KW}.data());
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnSetFilterNdDescriptor failed: " + std::string(cudnnGetErrorStringSafe(status)));
    }

    TensorDescriptorWrapper output_desc;
    output_desc.create();
    status = cudnnSetTensorNdDescriptor(output_desc.get(), CUDNN_DATA_FLOAT, 5,
                                        std::vector<int64_t>{N, C_out, D_out, H_out, W_out}.data());
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnSetTensorNdDescriptor output failed: " + std::string(cudnnGetErrorStringSafe(status)));
    }

    // Convolution descriptor
    ConvolutionDescriptorWrapper conv_desc;
    conv_desc.create();
    // Note: for 3D convolution, array lengths are 3 for padA, strideA, dilationA
    status = cudnnSetConvolutionNdDescriptor(conv_desc.get(), 3,
                                             padding_vec.data(),
                                             stride_vec.data(),
                                             dilation_vec.data(),
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnSetConvolutionNdDescriptor failed: " + std::string(cudnnGetErrorStringSafe(status)));
    }

    // Determine algorithm (use deterministic)
    cudnnConvolutionFwdAlgoPerf_t perf;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    // Simple heuristic: pick first viable algorithm
    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    status = cudnnFindConvolutionForwardAlgorithm(cudnn_handle,
                                                  input_desc.get(),
                                                  weight_desc.get(),
                                                  conv_desc.get(),
                                                  output_desc.get(),
                                                  8,
                                                  &returned_algo_count,
                                                  perf_results);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnFindConvolutionForwardAlgorithm failed: " + std::string(cudnnGetErrorStringSafe(status)));
    }
    if (returned_algo_count > 0) {
        algo = perf_results[0].algo;
    }

    // Allocate workspace
    size_t workspace_size = 0;
    status = cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle,
                                                     input_desc.get(),
                                                     weight_desc.get(),
                                                     conv_desc.get(),
                                                     output_desc.get(),
                                                     algo,
                                                     &workspace_size);
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error("cudnnGetConvolutionForwardWorkspaceSize failed: " + std::string(cudnnGetErrorStringSafe(status)));
    }
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t cuda_status = cudaMalloc(&workspace, workspace_size);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("cudaMalloc workspace failed");
        }
    }

    // Perform convolution
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (bias.has_value()) {
        status = cudnnConvolutionForward(cudnn_handle,
                                         &alpha,
                                         input_desc.get(), input_contig.data_ptr<float>(),
                                         weight_desc.get(), weight_contig.data_ptr<float>(),
                                         conv_desc.get(), algo,
                                         workspace, workspace_size,
                                         &beta,
                                         output_desc.get(), output.data_ptr<float>());
        if (status != CUDNN_STATUS_SUCCESS) {
            cudaFree(workspace);
            throw std::runtime_error("cudnnConvolutionForward failed: " + std::string(cudnnGetErrorStringSafe(status)));
        }
        // Add bias
        cudnnTensorDescriptor_t bias_desc;
        cudnnCreateTensorDescriptor(&bias_desc);
        cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                   1, static_cast<int>(C_out), 1, 1);
        const float beta_bias = 1.0f;
        status = cudnnAddTensor(cudnn_handle, &alpha, bias_desc, bias_contig.data_ptr<float>(),
                                &beta_bias, output_desc.get(), output.data_ptr<float>());
        cudnnDestroyTensorDescriptor(bias_desc);
        if (status != CUDNN_STATUS_SUCCESS) {
            cudaFree(workspace);
            throw std::runtime_error("cudnnAddTensor bias failed: " + std::string(cudnnGetErrorStringSafe(status)));
        }
    } else {
        status = cudnnConvolutionForward(cudnn_handle,
                                         &alpha,
                                         input_desc.get(), input_contig.data_ptr<float>(),
                                         weight_desc.get(), weight_contig.data_ptr<float>(),
                                         conv_desc.get(), algo,
                                         workspace, workspace_size,
                                         &beta,
                                         output_desc.get(), output.data_ptr<float>());
        if (status != CUDNN_STATUS_SUCCESS) {
            cudaFree(workspace);
            throw std::runtime_error("cudnnConvolutionForward failed: " + std::string(cudnnGetErrorStringSafe(status)));
        }
    }

    if (workspace != nullptr) {
        cudaFree(workspace);
    }

    return output;
}

// =============================================================================
// tanh_forward binding
// =============================================================================
torch::Tensor tanh_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "tanh_forward: input must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "tanh_forward: input must be float32");
    TORCH_CHECK(input.is_contiguous(), "tanh_forward: input must be contiguous");

    auto output = torch::empty_like(input);
    int64_t numel = input.numel();
    if (numel == 0) {
        return output;
    }
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    tanh_forward_launcher(input.data_ptr<float>(), output.data_ptr<float>(), numel, stream);
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "tanh_forward: CUDA kernel launch failed: ", cudaGetErrorString(err));
    return output;
}

// =============================================================================
// Registration
// =============================================================================
void register_custom_ops(pybind11::module& m) {
    m.def("conv3d_forward", &conv3d_forward,
          "3D convolution forward using cuDNN",
          pybind11::arg("input"),
          pybind11::arg("weight"),
          pybind11::arg("bias") = pybind11::none(),
          pybind11::arg("stride") = 1,
          pybind11::arg("padding") = 0,
          pybind11::arg("dilation") = 1);
    m.def("tanh_forward", &tanh_forward,
          "Element-wise tanh forward",
          pybind11::arg("input"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
