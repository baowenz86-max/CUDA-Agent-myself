#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <vector>
#include "../binding_registry.h"

// Declare launcher from custom_ops.cu
extern "C" void add_bias_launcher(
    const float* input,
    const float* bias,
    float* output,
    int batch,
    int channels,
    int length,
    long long in_stride_batch,
    long long in_stride_channel,
    long long in_stride_length,
    long long out_stride_batch,
    long long out_stride_channel,
    long long out_stride_length,
    cudaStream_t stream
);

static cudaStream_t getCurrentStream() {
    return c10::cuda::getCurrentCUDAStream().stream();
}

// Helper to create a torch::Tensor from raw pointer without any torch computational ops
// The tensor is created with the correct sizes and strides, and takes ownership of the pointer
// via a custom deleter that calls cudaFree.
// This is the only way to return a tensor without using torch::empty_like or torch::empty.
static torch::Tensor tensor_from_raw(
    void* data,
    c10::ScalarType dtype,
    const std::vector<int64_t>& sizes,
    const std::vector<int64_t>& strides,
    const c10::Device& device
) {
    auto options = torch::TensorOptions().dtype(dtype).device(device);
    auto tensor = torch::from_blob(
        data,
        c10::IntArrayRef(sizes),
        c10::IntArrayRef(strides),
        [](void* ptr) { cudaFree(ptr); },
        options
    );
    return tensor;
}

// Wrapper for add_bias: output = input + bias (broadcast over channels)
torch::Tensor add_bias(torch::Tensor input, torch::Tensor bias) {
    TORCH_CHECK(input.is_cuda(), "add_bias: input must be CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "add_bias: bias must be CUDA tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "add_bias: input must be float32");
    TORCH_CHECK(bias.scalar_type() == torch::kFloat32, "add_bias: bias must be float32");
    TORCH_CHECK(input.dim() == 3, "add_bias: input must be 3D (batch, channels, length)");
    TORCH_CHECK(bias.dim() == 2, "add_bias: bias must be 2D (channels, 1)");
    TORCH_CHECK(bias.size(0) == input.size(1), "add_bias: bias channels must match input channels");
    TORCH_CHECK(bias.size(1) == 1, "add_bias: bias second dimension must be 1");
    TORCH_CHECK(bias.is_contiguous(), "add_bias: bias must be contiguous");

    // Allocate output buffer with cudaMalloc
    int64_t numel = input.numel();
    if (numel == 0) {
        // Return empty tensor? Can't easily allocate zero bytes; return empty tensor with from_blob? Not supported.
        // In practice, input won't be empty for this task.
        TORCH_CHECK(false, "add_bias: empty input not supported");
    }
    float* output_ptr = nullptr;
    cudaError_t err = cudaMalloc(&output_ptr, numel * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "add_bias: cudaMalloc failed");

    // Get strides in elements
    auto strides = input.strides();
    long long in_stride_batch = strides[0];
    long long in_stride_channel = strides[1];
    long long in_stride_length = strides[2];

    int batch = input.size(0);
    int channels = input.size(1);
    int length = input.size(2);

    cudaStream_t stream = getCurrentStream();

    add_bias_launcher(
        input.data_ptr<float>(),
        bias.data_ptr<float>(),
        output_ptr,
        batch,
        channels,
        length,
        in_stride_batch,
        in_stride_channel,
        in_stride_length,
        in_stride_batch,  // output has same strides as input
        in_stride_channel,
        in_stride_length,
        stream
    );

    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "add_bias: CUDA kernel launch failed");

    // Create output tensor from raw pointer with same sizes and strides as input
    return tensor_from_raw(
        output_ptr,
        torch::kFloat32,
        input.sizes().vec(),
        input.strides().vec(),
        input.device()
    );
}

// Wrapper for conv_transpose1d_forward using cuDNN
torch::Tensor conv_transpose1d_forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias_opt,
    int64_t stride,
    int64_t padding,
    int64_t output_padding
) {
    TORCH_CHECK(input.is_cuda(), "conv_transpose1d_forward: input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "conv_transpose1d_forward: weight must be CUDA tensor");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->is_cuda(), "conv_transpose1d_forward: bias must be CUDA tensor");
    }
    TORCH_CHECK(input.scalar_type() == torch::kFloat32, "conv_transpose1d_forward: input must be float32");
    TORCH_CHECK(weight.scalar_type() == torch::kFloat32, "conv_transpose1d_forward: weight must be float32");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->scalar_type() == torch::kFloat32, "conv_transpose1d_forward: bias must be float32");
    }
    TORCH_CHECK(input.dim() == 3, "conv_transpose1d_forward: input must be 3D (N, C, L)");
    TORCH_CHECK(weight.dim() == 3, "conv_transpose1d_forward: weight must be 3D (C_in, C_out, kernel_size)");
    TORCH_CHECK(weight.is_contiguous(), "conv_transpose1d_forward: weight must be contiguous");

    // Input must be contiguous for cuDNN; if not, we need to make it contiguous via cudaMemcpy, not torch::contiguous
    // Since we cannot call input.contiguous(), we assume input is contiguous for now.
    // The contract says layout supports strides, but for cuDNN we need contiguous.
    // We can handle non-contiguous by checking is_contiguous and failing if not, or by copying manually.
    // For this task, the input from model_new.py is contiguous because it's the raw input tensor.
    TORCH_CHECK(input.is_contiguous(), "conv_transpose1d_forward: input must be contiguous for cuDNN");

    int64_t batch = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t in_length = input.size(2);
    int64_t out_channels = weight.size(1);
    int64_t kernel_size = weight.size(2);

    int64_t out_length = (in_length - 1) * stride - 2 * padding + kernel_size + output_padding;
    TORCH_CHECK(out_length > 0, "conv_transpose1d_forward: output length must be positive");
    int64_t core_out_length = out_length - output_padding; // actual output length from cuDNN without output_padding

    TORCH_CHECK(weight.size(0) == in_channels, "conv_transpose1d_forward: weight in_channels dimension mismatch");
    if (bias_opt.has_value()) {
        TORCH_CHECK(bias_opt->numel() == out_channels, "conv_transpose1d_forward: bias size must match out_channels");
    }

    // Create cuDNN handle
    cudnnHandle_t cudnn_handle;
    cudnnStatus_t status = cudnnCreate(&cudnn_handle);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnCreate failed");
    cudaStream_t stream = getCurrentStream();
    status = cudnnSetStream(cudnn_handle, stream);
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "conv_transpose1d_forward: cudnnSetStream failed");

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc, bias_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateTensorDescriptor(&bias_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    // Set input tensor descriptor: [batch, in_channels, 1, in_length]
    status = cudnnSetTensor4dDescriptor(
        input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, in_channels, 1, in_length
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "set input desc failed");

    // Set output tensor descriptor for core output (no output_padding): [batch, out_channels, 1, core_out_length]
    status = cudnnSetTensor4dDescriptor(
        output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        batch, out_channels, 1, core_out_length
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "set output desc failed");

    // Bias descriptor: [1, out_channels, 1, 1]
    status = cudnnSetTensor4dDescriptor(
        bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        1, out_channels, 1, 1
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "set bias desc failed");

    // Filter descriptor: weight in PyTorch is [in_channels, out_channels, kernel_size].
    // For cuDNN backward data, filter should be [out_channels, in_channels, 1, kernel_size].
    // We need to create a transposed weight in raw memory.
    // Since we cannot use torch::permute/contiguous, we will manually transpose the weight
    // using a custom CUDA kernel? Not allowed in .cpp. Instead, we can use cudaMemcpy2D to transpose?
    // However, cuDNN expects the filter in [K, C, R, S] contiguous layout.
    // We can use cudnnSetFilter4dDescriptor with a custom stride via cudnnSetFilter4dDescriptor_v4 if available,
    // but that is complex. For simplicity, we can allocate a new contiguous weight buffer
    // and use a loop on CPU? No, we are on GPU.
    // We will use a simple approach: since weight is small (3*16*3=144), we can copy it to CPU,
    // transpose on CPU, then copy back. But that involves CPU and is not allowed? Could be done with cudaMemcpy,
    // but we need to avoid torch computation. We can use cudaMemcpy and manual loops in C++ (not CUDA kernel).
    // Actually, we can use cudaMemcpy2DAsync to transpose the weight: treat weight as a 2D array of [in_channels, out_channels*kernel_size].
    // For each kernel element, we need to produce [out_channels, in_channels] matrix. We can use cudaMemcpy2DAsync with
    // source pitch = out_channels * kernel_size, destination pitch = in_channels * kernel_size, and width = out_channels * kernel_size,
    // height = in_channels, but that would copy the whole block including kernel dimension incorrectly.
    // A simpler method: allocate a new weight buffer on GPU with cudaMalloc, and manually copy/transpose using a CPU loop
    // with cudaMemcpyAsync for each element? That would be many small copies but doable for 144 elements.
    // Even simpler: we can read weight data to CPU using cudaMemcpy, transpose in CPU memory, then copy result back to GPU.
    // This is not a torch computation, just raw memory manipulation. Then pass to cuDNN.
    // Given weight size is tiny (144 floats = 576 bytes), this overhead is negligible.
    // We'll implement that.

    // Allocate CPU memory for weight and transposed weight
    float* weight_cpu = new float[in_channels * out_channels * kernel_size];
    float* weight_t_cpu = new float[out_channels * in_channels * kernel_size];
    // Copy weight from GPU to CPU
    cudaMemcpy(weight_cpu, weight.data_ptr<float>(), in_channels * out_channels * kernel_size * sizeof(float), cudaMemcpyDeviceToHost);
    // Transpose: for each in_channel, out_channel, kernel_idx
    // weight_cpu index: (in_c * out_channels + out_c) * kernel_size + k
    // weight_t_cpu index: (out_c * in_channels + in_c) * kernel_size + k
    for (int64_t in_c = 0; in_c < in_channels; ++in_c) {
        for (int64_t out_c = 0; out_c < out_channels; ++out_c) {
            for (int64_t k = 0; k < kernel_size; ++k) {
                int64_t src_idx = (in_c * out_channels + out_c) * kernel_size + k;
                int64_t dst_idx = (out_c * in_channels + in_c) * kernel_size + k;
                weight_t_cpu[dst_idx] = weight_cpu[src_idx];
            }
        }
    }
    // Allocate GPU memory for transposed weight
    float* weight_t_gpu = nullptr;
    cudaMalloc(&weight_t_gpu, out_channels * in_channels * kernel_size * sizeof(float));
    cudaMemcpy(weight_t_gpu, weight_t_cpu, out_channels * in_channels * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    delete[] weight_cpu;
    delete[] weight_t_cpu;

    // Set filter descriptor with transposed weight layout: [out_channels, in_channels, 1, kernel_size]
    status = cudnnSetFilter4dDescriptor(
        filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        out_channels, in_channels, 1, kernel_size
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "set filter desc failed");

    // Convolution descriptor for transposed conv (backward data)
    // We set padding, stride, dilation=1, mode=CROSS_CORRELATION
    status = cudnnSetConvolution2dDescriptor(
        conv_desc,
        0,             // pad_h (height=1, no padding in height)
        padding,       // pad_w
        stride,        // stride_h
        stride,        // stride_w
        1, 1,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "set conv desc failed");

    // Allocate workspace
    size_t workspace_size = 0;
    status = cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle,
        filter_desc,
        input_desc,
        conv_desc,
        output_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        &workspace_size
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "get workspace size failed");
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    // Allocate core output buffer
    float* core_output = nullptr;
    cudaMalloc(&core_output, batch * out_channels * core_out_length * sizeof(float));

    float alpha = 1.0f, beta = 0.0f;
    status = cudnnConvolutionBackwardData(
        cudnn_handle,
        &alpha,
        filter_desc,
        weight_t_gpu,
        input_desc,
        input.data_ptr<float>(),
        conv_desc,
        CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
        workspace,
        workspace_size,
        &beta,
        output_desc,
        core_output
    );
    TORCH_CHECK(status == CUDNN_STATUS_SUCCESS, "cudnnConvolutionBackwardData failed");

    // Allocate final output buffer with full length (including output_padding)
    float* final_output = nullptr;
    cudaMalloc(&final_output, batch * out_channels * out_length * sizeof(float));
    // Zero entire final_output
    cudaMemsetAsync(final_output, 0, batch * out_channels * out_length * sizeof(float), stream);

    // Copy core_output into final_output for each (batch, channel) as a row of length core_out_length
    // We use cudaMemcpy2DAsync: treat height = batch*out_channels, width = core_out_length,
    // source pitch = core_out_length * sizeof(float), destination pitch = out_length * sizeof(float)
    cudaMemcpy2DAsync(
        final_output, out_length * sizeof(float),
        core_output, core_out_length * sizeof(float),
        core_out_length * sizeof(float),
        batch * out_channels,
        cudaMemcpyDeviceToDevice,
        stream
    );

    // Add bias if provided (in-place on final_output)
    if (bias_opt.has_value()) {
        // bias is 1D contiguous of size out_channels
        // We call add_bias_launcher directly with final_output as input/output and bias pointer
        add_bias_launcher(
            final_output,
            bias_opt->data_ptr<float>(),
            final_output,
            batch,
            out_channels,
            out_length,
            out_length, // stride batch? Actually final_output is contiguous: stride_batch = out_channels*out_length, stride_channel = out_length, stride_length = 1
            (long long)out_channels * out_length,
            out_length,
            1,
            (long long)out_channels * out_length,
            out_length,
            1,
            stream
        );
    }

    // Clean up cuDNN resources
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyTensorDescriptor(bias_desc);
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn_handle);
    if (workspace) cudaFree(workspace);
    cudaFree(weight_t_gpu);
    cudaFree(core_output);

    // Create output tensor from raw pointer with contiguous layout
    return tensor_from_raw(
        final_output,
        torch::kFloat32,
        {batch, out_channels, out_length},
        {out_channels * out_length, out_length, 1},
        input.device()
    );
}

// Registration
void register_custom_ops(pybind11::module& m) {
    m.def("conv_transpose1d_forward", &conv_transpose1d_forward,
          "ConvTranspose1d forward using cuDNN",
          py::arg("input"), py::arg("weight"), py::arg("bias"),
          py::arg("stride"), py::arg("padding"), py::arg("output_padding"));
    m.def("add_bias", &add_bias, "Add bias with broadcast",
          py::arg("input"), py::arg("bias"));
}

REGISTER_BINDING(custom_ops, register_custom_ops);
