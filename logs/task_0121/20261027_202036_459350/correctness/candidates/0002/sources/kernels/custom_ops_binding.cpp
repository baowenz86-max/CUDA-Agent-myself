// Custom binding for pad1d, conv_transpose2d (cuDNN via dynamic loading), add_bias
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <dlfcn.h>
#include <vector>
#include <stdexcept>
#include <string>
#include "../binding_registry.h"

// Launcher declarations from custom_ops.cu
extern "C" void custom_pad1d_forward_launcher(
    const float* x, float* out,
    int B, int C, int H, int W,
    int pad_left, int pad_right,
    cudaStream_t stream);

extern "C" void custom_add_bias_forward_launcher(
    const float* x, const float* bias, float* out,
    int B, int C, int H, int W,
    cudaStream_t stream);

// Helper to check CUDA errors
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err)); \
    } \
}

// ==================== pad1d ====================
torch::Tensor custom_pad1d_forward(
    torch::Tensor x,
    int64_t padding_left,
    int64_t padding_right)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);
    int64_t out_W = W + padding_left + padding_right;

    auto out = torch::empty({B, C, H, out_W}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_pad1d_forward_launcher(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        static_cast<int>(padding_left),
        static_cast<int>(padding_right),
        stream
    );

    CHECK_CUDA(cudaGetLastError());
    return out;
}

// ==================== conv_transpose2d (cuDNN dynamic loading) ====================
// Define necessary cuDNN types and function pointer types (minimal subset)
typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnConvolutionDescriptor_t;

typedef enum {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
    CUDNN_STATUS_LICENSE_ERROR = 10,
    CUDNN_STATUS_RUNTIME_PREREQUISITE_MISSING = 11,
    CUDNN_STATUS_RUNTIME_IN_PROGRESS = 12,
    CUDNN_STATUS_RUNTIME_FP_OVERFLOW = 13
} cudnnStatus_t;

typedef enum {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2
} cudnnTensorFormat_t;

typedef enum {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8
} cudnnDataType_t;

typedef enum {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1
} cudnnConvolutionMode_t;

typedef enum {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_FFT = 2,
    CUDNN_CONVOLUTION_BWD_DATA_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_DATA_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT = 6
} cudnnConvolutionBwdDataAlgo_t;

// Function pointer types for used cuDNN APIs
typedef cudnnStatus_t (*cudnnCreate_t)(cudnnHandle_t*);
typedef cudnnStatus_t (*cudnnSetStream_t)(cudnnHandle_t, cudaStream_t);
typedef cudnnStatus_t (*cudnnCreateTensorDescriptor_t)(cudnnTensorDescriptor_t*);
typedef cudnnStatus_t (*cudnnSetTensor4dDescriptor_t)(cudnnTensorDescriptor_t, cudnnTensorFormat_t, cudnnDataType_t, int, int, int, int);
typedef cudnnStatus_t (*cudnnCreateFilterDescriptor_t)(cudnnFilterDescriptor_t*);
typedef cudnnStatus_t (*cudnnSetFilter4dDescriptor_t)(cudnnFilterDescriptor_t, cudnnDataType_t, cudnnTensorFormat_t, int, int, int, int);
typedef cudnnStatus_t (*cudnnCreateConvolutionDescriptor_t)(cudnnConvolutionDescriptor_t*);
typedef cudnnStatus_t (*cudnnSetConvolution2dDescriptor_t)(cudnnConvolutionDescriptor_t, int, int, int, int, int, int, cudnnConvolutionMode_t, cudnnDataType_t);
typedef cudnnStatus_t (*cudnnGetConvolutionBackwardDataWorkspaceSize_t)(cudnnHandle_t, cudnnFilterDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionDescriptor_t, cudnnTensorDescriptor_t, cudnnConvolutionBwdDataAlgo_t, size_t*);
typedef cudnnStatus_t (*cudnnConvolutionBackwardData_t)(cudnnHandle_t, const void*, cudnnFilterDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t, cudnnConvolutionBwdDataAlgo_t, void*, size_t, const void*, cudnnTensorDescriptor_t, void*);
typedef cudnnStatus_t (*cudnnDestroyTensorDescriptor_t)(cudnnTensorDescriptor_t);
typedef cudnnStatus_t (*cudnnDestroyFilterDescriptor_t)(cudnnFilterDescriptor_t);
typedef cudnnStatus_t (*cudnnDestroyConvolutionDescriptor_t)(cudnnConvolutionDescriptor_t);

// Global function pointers, loaded once via dlopen
static void* cudnn_lib = nullptr;
static cudnnCreate_t cudnnCreate_fn = nullptr;
static cudnnSetStream_t cudnnSetStream_fn = nullptr;
static cudnnCreateTensorDescriptor_t cudnnCreateTensorDescriptor_fn = nullptr;
static cudnnSetTensor4dDescriptor_t cudnnSetTensor4dDescriptor_fn = nullptr;
static cudnnCreateFilterDescriptor_t cudnnCreateFilterDescriptor_fn = nullptr;
static cudnnSetFilter4dDescriptor_t cudnnSetFilter4dDescriptor_fn = nullptr;
static cudnnCreateConvolutionDescriptor_t cudnnCreateConvolutionDescriptor_fn = nullptr;
static cudnnSetConvolution2dDescriptor_t cudnnSetConvolution2dDescriptor_fn = nullptr;
static cudnnGetConvolutionBackwardDataWorkspaceSize_t cudnnGetConvolutionBackwardDataWorkspaceSize_fn = nullptr;
static cudnnConvolutionBackwardData_t cudnnConvolutionBackwardData_fn = nullptr;
static cudnnDestroyTensorDescriptor_t cudnnDestroyTensorDescriptor_fn = nullptr;
static cudnnDestroyFilterDescriptor_t cudnnDestroyFilterDescriptor_fn = nullptr;
static cudnnDestroyConvolutionDescriptor_t cudnnDestroyConvolutionDescriptor_fn = nullptr;

// Load cuDNN symbols once
static void load_cudnn_symbols() {
    if (cudnn_lib != nullptr) return; // already loaded
    // Try common library names
    const char* libnames[] = {"libcudnn.so", "libcudnn.so.8", "libcudnn.so.7", nullptr};
    for (int i = 0; libnames[i] != nullptr; ++i) {
        cudnn_lib = dlopen(libnames[i], RTLD_LAZY | RTLD_GLOBAL);
        if (cudnn_lib) break;
    }
    if (!cudnn_lib) {
        throw std::runtime_error("Failed to load cuDNN library. Please ensure cuDNN is installed and in library path.");
    }
    // Load function pointers
    #define LOAD_SYMBOL(name) \
        name##_fn = reinterpret_cast<name##_t>(dlsym(cudnn_lib, #name)); \
        if (!name##_fn) { \
            dlclose(cudnn_lib); \
            cudnn_lib = nullptr; \
            throw std::runtime_error("Failed to load cuDNN symbol: " #name); \
        }
    LOAD_SYMBOL(cudnnCreate);
    LOAD_SYMBOL(cudnnSetStream);
    LOAD_SYMBOL(cudnnCreateTensorDescriptor);
    LOAD_SYMBOL(cudnnSetTensor4dDescriptor);
    LOAD_SYMBOL(cudnnCreateFilterDescriptor);
    LOAD_SYMBOL(cudnnSetFilter4dDescriptor);
    LOAD_SYMBOL(cudnnCreateConvolutionDescriptor);
    LOAD_SYMBOL(cudnnSetConvolution2dDescriptor);
    LOAD_SYMBOL(cudnnGetConvolutionBackwardDataWorkspaceSize);
    LOAD_SYMBOL(cudnnConvolutionBackwardData);
    LOAD_SYMBOL(cudnnDestroyTensorDescriptor);
    LOAD_SYMBOL(cudnnDestroyFilterDescriptor);
    LOAD_SYMBOL(cudnnDestroyConvolutionDescriptor);
    #undef LOAD_SYMBOL
}

// Helper macro to call cuDNN functions via pointers and check status
#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        throw std::runtime_error(std::string("cuDNN error with status code: ") + std::to_string(static_cast<int>(status))); \
    } \
}

// ConvTranspose2d implementation using cuDNN backward data
torch::Tensor custom_conv_transpose2d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t stride,
    int64_t output_padding)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    // bias can be undefined
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor if defined");
    }
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    if (bias.defined()) {
        TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32 if defined");
    }
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    if (bias.defined()) {
        TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous if defined");
    }

    // Load cuDNN symbols if not already loaded
    load_cudnn_symbols();

    int64_t B = x.size(0);
    int64_t C_in = x.size(1);
    int64_t H_in = x.size(2);
    int64_t W_in = x.size(3);
    int64_t C_out = weight.size(1);
    int64_t kH = weight.size(2);
    int64_t kW = weight.size(3);

    // Output dimensions for ConvTranspose2d with stride and output_padding
    int64_t H_out = (H_in - 1) * stride + (kH - 1) + 1 + output_padding;
    int64_t W_out = (W_in - 1) * stride + (kW - 1) + 1 + output_padding;

    auto out = torch::empty({B, C_out, H_out, W_out}, x.options());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    // Create cuDNN handle (static, but we need to ensure it's created after load)
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        CHECK_CUDNN((*cudnnCreate_fn)(&handle));
    }
    CHECK_CUDNN((*cudnnSetStream_fn)(handle, stream));

    // Descriptors
    cudnnTensorDescriptor_t x_desc = nullptr;
    CHECK_CUDNN((*cudnnCreateTensorDescriptor_fn)(&x_desc));
    CHECK_CUDNN((*cudnnSetTensor4dDescriptor_fn)(
        x_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        static_cast<int>(B),
        static_cast<int>(C_in),
        static_cast<int>(H_in),
        static_cast<int>(W_in)
    ));

    cudnnFilterDescriptor_t w_desc = nullptr;
    CHECK_CUDNN((*cudnnCreateFilterDescriptor_fn)(&w_desc));
    CHECK_CUDNN((*cudnnSetFilter4dDescriptor_fn)(
        w_desc,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        static_cast<int>(C_in),
        static_cast<int>(C_out),
        static_cast<int>(kH),
        static_cast<int>(kW)
    ));

    cudnnTensorDescriptor_t y_desc = nullptr;
    CHECK_CUDNN((*cudnnCreateTensorDescriptor_fn)(&y_desc));
    CHECK_CUDNN((*cudnnSetTensor4dDescriptor_fn)(
        y_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        static_cast<int>(B),
        static_cast<int>(C_out),
        static_cast<int>(H_out),
        static_cast<int>(W_out)
    ));

    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    CHECK_CUDNN((*cudnnCreateConvolutionDescriptor_fn)(&conv_desc));
    // For ConvTranspose2d, padding is 0 in the underlying cuDNN call (as PyTorch does not set padding for this module)
    CHECK_CUDNN((*cudnnSetConvolution2dDescriptor_fn)(
        conv_desc,
        0, 0, // pad_h, pad_w
        static_cast<int>(stride), static_cast<int>(stride),
        1, 1, // dilation_h, dilation_w
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));

    // Use a fixed algorithm (could use get algo for performance, but this is a repair)
    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    size_t workspace_size = 0;
    CHECK_CUDNN((*cudnnGetConvolutionBackwardDataWorkspaceSize_fn)(
        handle, w_desc, x_desc, conv_desc, y_desc, algo, &workspace_size));

    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN((*cudnnConvolutionBackwardData_fn)(
        handle,
        &alpha,
        w_desc, weight.data_ptr<float>(),
        x_desc, x.data_ptr<float>(),
        conv_desc, algo,
        workspace, workspace_size,
        &beta,
        y_desc, out.data_ptr<float>()
    ));

    // Add bias if defined
    if (bias.defined()) {
        custom_add_bias_forward_launcher(
            out.data_ptr<float>(),
            bias.data_ptr<float>(),
            out.data_ptr<float>(),
            static_cast<int>(B),
            static_cast<int>(C_out),
            static_cast<int>(H_out),
            static_cast<int>(W_out),
            stream
        );
    }

    // Cleanup descriptors
    CHECK_CUDNN((*cudnnDestroyTensorDescriptor_fn)(x_desc));
    CHECK_CUDNN((*cudnnDestroyFilterDescriptor_fn)(w_desc));
    CHECK_CUDNN((*cudnnDestroyTensorDescriptor_fn)(y_desc));
    CHECK_CUDNN((*cudnnDestroyConvolutionDescriptor_fn)(conv_desc));
    if (workspace) {
        CHECK_CUDA(cudaFree(workspace));
    }

    CHECK_CUDA(cudaGetLastError());
    return out;
}

// ==================== add_bias ====================
torch::Tensor custom_add_bias_forward(
    torch::Tensor x,
    torch::Tensor bias)
{
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");

    int64_t B = x.size(0);
    int64_t C = x.size(1);
    int64_t H = x.size(2);
    int64_t W = x.size(3);

    auto out = torch::empty_like(x);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    custom_add_bias_forward_launcher(
        x.data_ptr<float>(),
        bias.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(B),
        static_cast<int>(C),
        static_cast<int>(H),
        static_cast<int>(W),
        stream
    );

    CHECK_CUDA(cudaGetLastError());
    return out;
}

// ==================== Register bindings ====================
void register_custom_ops(pybind11::module& m) {
    m.def("custom_pad1d_forward", &custom_pad1d_forward,
          "ConstantPad1d forward (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("padding_left"),
          pybind11::arg("padding_right"));

    m.def("custom_conv_transpose2d_forward", &custom_conv_transpose2d_forward,
          "ConvTranspose2d forward using cuDNN (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("weight"),
          pybind11::arg("bias"),
          pybind11::arg("stride"),
          pybind11::arg("output_padding"));

    m.def("custom_add_bias_forward", &custom_add_bias_forward,
          "Add bias with shape [C,1,1] to x (CUDA)",
          pybind11::arg("x"),
          pybind11::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
