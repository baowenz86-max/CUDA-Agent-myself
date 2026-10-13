#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>

#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cudnn.h>

#include <vector>
#include <stdexcept>

#include "../binding_registry.h"

// Declare launchers from custom_ops.cu
extern "C" void bias_add_forward_launcher(
    float* out,
    const float* in,
    const float* bias,
    int batch,
    int channels,
    int length,
    int batch_stride,
    int channel_stride,
    int inner_stride,
    cudaStream_t stream);

extern "C" void round_elementwise_forward_launcher(
    float* out,
    const float* in,
    int ndims,
    const int* shape,
    const int* strides,
    int total_elements,
    cudaStream_t stream);

// Helper to get cuDNN handle for current device
typedef struct CudnnHandle {
    cudnnHandle_t handle;
    CudnnHandle() {
        cudnnCreate(&handle);
    }
    ~CudnnHandle() {
        cudnnDestroy(handle);
    }
} CudnnHandle;

static cudnnHandle_t get_cudnn_handle() {
    static thread_local CudnnHandle holder;
    return holder.handle;
}

// ============ ConvTranspose1d forward using cuDNN ============
torch::Tensor conv_transpose1d_forward(
    torch::Tensor x,
    torch::Tensor weight,
    int64_t stride,
    int64_t padding,
    int64_t output_padding) {

    // Input validation
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(weight.dtype() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D (batch, channels, length)");
    TORCH_CHECK(weight.dim() == 3, "weight must be 3D (in_channels, out_channels, kernel_size)");

    int64_t batch = x.size(0);
    int64_t in_channels = x.size(1);
    int64_t length_in = x.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    TORCH_CHECK(weight.size(0) == in_channels, "weight in_channels mismatch");
    TORCH_CHECK(stride > 0, "stride must be positive");
    TORCH_CHECK(padding >= 0, "padding must be non-negative");
    TORCH_CHECK(output_padding >= 0, "output_padding must be non-negative");
    TORCH_CHECK(output_padding < stride, "output_padding must be less than stride");

    // Compute output length
    int64_t length_out = (length_in - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(length_out > 0, "output length must be positive");

    // Allocate output tensor (contiguous, same device and dtype as input)
    auto out = torch::empty({batch, out_channels, length_out}, x.options());

    // Ensure input and weight are contiguous
    auto x_contig = x.contiguous();
    auto weight_contig = weight.contiguous();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudnnHandle_t handle = get_cudnn_handle();
    cudnnSetStream(handle, stream);

    // Create tensor descriptors
    cudnnTensorDescriptor_t x_desc, y_desc;
    cudnnFilterDescriptor_t w_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&x_desc);
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnCreateFilterDescriptor(&w_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // For ConvTranspose1d, cuDNN expects 2D or 3D but can use NCHW with height=1.
    // We treat length as width and set height=1.
    int x_dims[4] = {(int)batch, (int)in_channels, 1, (int)length_in};
    int x_strides[4] = {(int)(in_channels * 1 * length_in), (int)(1 * length_in), (int)length_in, 1};
    cudnnSetTensor4dDescriptor(x_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    // Note: cuDNN 8 uses cudnnSetTensor4dDescriptor; strides are implicit for NCHW.
    // Actually cudnnSetTensor4dDescriptor sets strides automatically.

    int y_dims[4] = {(int)batch, (int)out_channels, 1, (int)length_out};
    cudnnSetTensor4dDescriptor(y_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                               y_dims[0], y_dims[1], y_dims[2], y_dims[3]);

    // Filter descriptor: weight shape [in_channels, out_channels, kernel_size] -> cuDNN filter layout: KCRS or something? For convolution, filter dims: [out_channels, in_channels, kernel_h, kernel_w]. For transposed conv (backward data), weight shape is [in_channels, out_channels, kernel_size], cuDNN expects filter for convDesc as [out_channels, in_channels/groups, kernel_h, kernel_w]. For transposed, it's the same filter but used in backward data. Weight tensor in PyTorch is [in_channels, out_channels, kernel_size], so for cuDNN we need to set filter as [out_channels, in_channels, kernel_size]? Actually in cuDNN, filter tensor dims are [K, C, R, S] where K = number of output feature maps, C = number of input feature maps, R, S = kernel dimensions. For PyTorch ConvTranspose1d weight is [in_channels, out_channels, kernel_size], i.e., [C, K, R]. So we can treat it as a 3D filter but cuDNN expects 4D [K, C, R, S] with S=1. So we need to permute? But the underlying data is contiguous [C, K, R]. cuDNN expects [K, C, R, S] i.e., [K, C, R, 1]. We cannot permute without copying or using stride? We can create filter descriptor with dimensions [out_channels, in_channels, 1, kernel_size]? No, filter dims are [K, C, R, S] where R and S are spatial. So R = kernel_size, S = 1. The memory layout should be K-major, then C, then R, then S. Our weight is C-major (in_channels outer, out_channels middle, kernel_size inner). That is the transpose. For ConvTranspose, PyTorch stores weight as [in_channels, out_channels, kernel_size]. cuDNN's backward data filter descriptor expects [out_channels, in_channels, kernel_size, 1]. So we need to set filter stride accordingly.
    // We'll use cudnnSetFilter4dDescriptor with appropriate strides.
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               (int)out_channels, (int)in_channels, (int)kernel_size, 1);
    // But the actual memory layout is [in_channels, out_channels, kernel_size], so we need to tell cuDNN that the strides are non-standard. By default, cudnnSetFilter4dDescriptor sets strides in NCHW order? Actually filter descriptor does not have strides; it assumes contiguous in the specified order. We need to use cudnnSetFilter4dDescriptor with explicit strides? There is cudnnSetFilter4dDescriptor_v4? In cuDNN v8, there is cudnnSetFilter4dDescriptor, which sets the descriptor internal format to CUDNN_TENSOR_NCHW and assumes contiguous. The weight tensor from PyTorch is contiguous [in_channels, out_channels, kernel_size]. This is not the expected order [out_channels, in_channels, kernel_size]. To avoid permuting, we can treat the weight as a 3D filter using cudnnSetFilterNdDescriptor with 3 dimensions? cuDNN supports 3D tensor? For 1D conv, cuDNN requires 2D or 3D spatial, so we use 2D with height=1. For filter, we need 4D [K, C, R, S] with S=1. So we must permute weight to [out_channels, in_channels, kernel_size] before passing? That would require a copy. Alternatively, we can set the filter descriptor with explicit strides to account for the non-standard layout. cuDNN API has cudnnSetFilter4dDescriptor_v4 with strides, but that's deprecated. Better: use cudnnSetFilterNdDescriptor with CUDNN_TENSOR_NCHW and dims [out_channels, in_channels, kernel_size, 1] but tell it the tensor is actually transposed by setting the strides? Actually the descriptor only describes logical dimensions, and the data pointer is assumed to be in that logical order. So we need to physically have the weight in [out_channels, in_channels, kernel_size] order. We can create a temporary permuted tensor: auto weight_permuted = weight_contig.permute({1,0,2}).contiguous(); That will allocate and copy. Since the weight is small (4*8*3 = 96 elements), it's negligible. Then pass weight_permuted.data_ptr<float>().

    auto weight_permuted = weight_contig.permute({1, 0, 2}).contiguous();

    int filter_dims[4] = {(int)out_channels, (int)in_channels, (int)kernel_size, 1};
    cudnnSetFilter4dDescriptor(w_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                               filter_dims[0], filter_dims[1], filter_dims[2], filter_dims[3]);

    // Convolution descriptor for backward data (transposed conv).
    // For ConvTranspose, padding, stride, output_padding are mirrored.
    cudnnSetConvolution2dDescriptor(conv_desc,
                                    padding, padding,  // pad_h, pad_w (same as padding)
                                    stride, stride,    // stride_h, stride_w
                                    1, 1,              // dilation_h, dilation_w
                                    CUDNN_CROSS_CORRELATION, // mode: cross correlation matches PyTorch
                                    CUDNN_DATA_FLOAT);
    // Note: output_padding is not directly supported in cuDNN; it only affects output size. We already set output size accordingly. The extra output positions will be zero-padded automatically because the backward data operation with output_padding > 0 effectively means the last output_padding columns are not computed from any input, so they are zero. cuDNN's backward data with given conv descriptor computes output of size (input_size-1)*stride - 2*pad + kernel_size. If we request a larger output tensor (length_out with output_padding), cuDNN will only write to the first part, leaving the rest uninitialized. We must explicitly zero the extra columns. Since output_padding is 1, the last column of each output channel will be zero. We'll zero initialize the output tensor before cuDNN call.

    cudnnSetStream(handle, stream);

    // Zero out the extra output_padding columns (if any). For simplicity, zero entire output, then cuDNN writes the valid part.
    // Actually we need to zero only the extra columns, but zeroing whole output is fine even if overwritten.
    cudaMemsetAsync(out.data_ptr<float>(), 0, batch * out_channels * length_out * sizeof(float), stream);

    // Perform backward data (transposed convolution)
    float alpha = 1.0f, beta = 0.0f; // beta 0 after zeroing; or we could use beta=0 and no zeroing. Since we zeroed, use beta=1? Use beta=0 and no zeroing, but then output_padding positions uninitialized. We'll use beta=0, and later zero the extra positions manually? We already zeroed entire output, so beta=0 will overwrite with computed values where valid, and leave zeros elsewhere? No, beta=0 does not read the output; it writes the result to output. The positions not written by cuDNN (if any) remain whatever was there from zeroing. So zeroing prior and beta=0 is fine.
    cudnnConvolutionBackwardData(handle,
                                 &alpha,
                                 w_desc, weight_permuted.data_ptr<float>(),
                                 x_desc, x_contig.data_ptr<float>(),
                                 conv_desc,
                                 CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, // choose default; maybe use cudnnGetConvolutionBackwardDataAlgorithm
                                 nullptr, 0,
                                 &beta,
                                 y_desc, out.data_ptr<float>());

    // Check for errors
    cudnnStatus_t status = cudnnGetLastError? There is no cudnnGetLastError. Use cudnnGetErrorString.
    // Actually check return of cudnnConvolutionBackwardData

    cudnnDestroyTensorDescriptor(x_desc);
    cudnnDestroyTensorDescriptor(y_desc);
    cudnnDestroyFilterDescriptor(w_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    return out;
}

// ============ Bias add forward ============
torch::Tensor bias_add_forward(torch::Tensor x, torch::Tensor bias) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(bias.dtype() == torch::kFloat32, "bias must be float32");
    TORCH_CHECK(x.dim() == 3, "x must be 3D");
    TORCH_CHECK(bias.dim() == 1, "bias must be 1D");

    int64_t batch = x.size(0);
    int64_t channels = x.size(1);
    int64_t length = x.size(2);
    TORCH_CHECK(bias.size(0) == channels, "bias size must match channels");

    auto bias_contig = bias.contiguous();
    auto out = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int64_t batch_stride = x.stride(0);
    int64_t channel_stride = x.stride(1);
    int64_t inner_stride = x.stride(2);

    int total_elements = batch * channels * length;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    bias_add_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        bias_contig.data_ptr<float>(),
        (int)batch,
        (int)channels,
        (int)length,
        (int)batch_stride,
        (int)channel_stride,
        (int)inner_stride,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "bias_add_forward kernel launch failed: ", cudaGetErrorString(err));

    return out;
}

// ============ Round elementwise forward ============
torch::Tensor round_elementwise_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");

    auto out = torch::empty_like(x, x.options().memory_format(torch::MemoryFormat::Contiguous));

    int ndims = x.dim();
    int total_elements = x.numel();

    std::vector<int> shape_cpu(ndims);
    std::vector<int> strides_cpu(ndims);
    for (int i = 0; i < ndims; ++i) {
        shape_cpu[i] = (int)x.size(i);
        strides_cpu[i] = (int)x.stride(i);
    }

    int* shape_dev = nullptr;
    int* strides_dev = nullptr;
    cudaMalloc(&shape_dev, ndims * sizeof(int));
    cudaMalloc(&strides_dev, ndims * sizeof(int));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    cudaMemcpyAsync(shape_dev, shape_cpu.data(), ndims * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(strides_dev, strides_cpu.data(), ndims * sizeof(int), cudaMemcpyHostToDevice, stream);

    round_elementwise_forward_launcher(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        ndims,
        shape_dev,
        strides_dev,
        total_elements,
        stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "round_elementwise_forward kernel launch failed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaFree(shape_dev);
    cudaFree(strides_dev);

    return out;
}

// ============ Registration ============
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "1D transposed convolution without bias",
          py::arg("x"), py::arg("weight"), py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("bias_add_forward", &bias_add_forward,
          "Add bias to each channel of a 3D tensor",
          py::arg("x"), py::arg("bias"));
    m.def("round_elementwise_forward", &round_elementwise_forward,
          "Round elements to nearest integer, half to even",
          py::arg("x"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
