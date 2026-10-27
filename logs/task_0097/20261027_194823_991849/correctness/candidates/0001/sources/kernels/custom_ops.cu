// kernels/custom_ops.cu
// Contains CUDA kernels and launchers for custom_conv_transpose1d and custom_histc.
// This file must not include any torch/ATen/pybind headers.
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>
#include <cmath>

// -----------------------------------------------------------------------------
// Histogram kernel: computes histogram of a float tensor with given bins, min, max.
// The histogram is computed by atomically incrementing counts per bin.
// Input: flat array of size total_elements.
// Output: hist array of size bins, initialized to zero.
// Kernel uses grid-stride loop to handle arbitrarily large inputs.
// -----------------------------------------------------------------------------
__global__ void histc_kernel(const float* __restrict__ input,
                             float* __restrict__ output,
                             int total_elements,
                             int bins,
                             float min_val,
                             float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float bin_width = (max_val - min_val) / bins;

    for (int i = idx; i < total_elements; i += stride) {
        float val = input[i];
        if (val >= min_val && val <= max_val) {
            int bin = (int)((val - min_val) / bin_width);
            if (bin == bins) bin = bins - 1; // include max_val in last bin
            if (bin >= 0 && bin < bins) {
                atomicAdd(&output[bin], 1.0f);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Host launcher for custom_histc.
// Parameters:
//   input: pointer to input tensor data (float32, contiguous)
//   output: pointer to output tensor data (float32, size bins)
//   total_elements: number of elements in input
//   bins: number of histogram bins
//   min_val, max_val: histogram range
//   stream: CUDA stream to execute on
// -----------------------------------------------------------------------------
extern "C" void custom_histc_launcher(const float* input,
                                      float* output,
                                      int total_elements,
                                      int bins,
                                      float min_val,
                                      float max_val,
                                      cudaStream_t stream) {
    // Initialize output to zero
    cudaMemsetAsync(output, 0, bins * sizeof(float), stream);

    // Launch configuration
    int block_size = 256;
    int num_blocks = (total_elements + block_size - 1) / block_size;
    // Cap grid size to avoid exceeding max grid dimension
    const int max_grid_size_x = 65535; // common limit, adjust if needed
    if (num_blocks > max_grid_size_x) {
        num_blocks = max_grid_size_x;
    }

    if (total_elements > 0) {
        histc_kernel<<<num_blocks, block_size, 0, stream>>>(
            input, output, total_elements, bins, min_val, max_val);
    }

    // Error check can be added by caller if needed, but we can check here:
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) { ... }
}

// -----------------------------------------------------------------------------
// ConvTranspose1d via cuDNN
// We implement a host launcher that uses cuDNN to perform the transposed convolution.
// This is the only place where cuDNN is used, as required.
// -----------------------------------------------------------------------------

// Helper to create a cuDNN tensor descriptor for a 4D tensor (N, C, H, W) with 1D data.
// For Conv1d, we treat the length dimension as width (W) and set height=1.
// Thus input shape: [N, C, 1, L], output shape: [N, C, 1, L_out].
// Filter shape for ConvTranspose: [C_in, C_out, 1, kernel_size] (cuDNN expects filter in [k, c, r, s]? Actually for conv_transpose, cuDNN descriptor follows convolution where filter dims are [C_out, C_in, kernel_h, kernel_w]? Need careful).
// We'll use the cuDNN API: cudnnConvolutionBackwardData or cudnnConvolutionForward with transposed filter? PyTorch's ConvTranspose1d is implemented as "deconvolution" using cudnnConvolutionBackwardData.
// We'll follow PyTorch's approach: use cudnnConvolutionBackwardData with appropriate parameters.
// For simplicity, we can also use cudnnConvolutionForward with filter arranged appropriately, but cudnnConvolutionBackwardData is directly intended for transposed conv.
// However, we need to construct descriptors carefully.
// The function signature: cudnnConvolutionBackwardData(handle, alpha, wDesc, w, dyDesc, dy, convDesc, beta, dxDesc, dx)
// wDesc describes the filter of the forward convolution (which is the same as our weight for conv_transpose? Actually in PyTorch, for ConvTranspose, weight shape is [in_channels, out_channels, kernel_size] and it's used as the filter in the backward data pass.
// We'll define descriptors accordingly.

// We need to keep a global cuDNN handle or create per call. Creating per call is expensive but safe; we can use a static handle initialized once.
static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (handle == nullptr) {
        cudnnCreate(&handle);
    }
    return handle;
}

// Function to check cuDNN errors
#define CHECK_CUDNN(expr) \
    do { \
        cudnnStatus_t status = (expr); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, cudnnGetErrorString(status)); \
            return; \
        } \
    } while (0)

// Launcher for custom_conv_transpose1d.
// Parameters:
//   input: pointer to input tensor (N, C_in, L_in)
//   weight: pointer to weight (C_in, C_out/groups, kernel_size) (for groups=1 here, so C_in, C_out, kernel_size)
//   bias: pointer to bias (C_out) or nullptr if no bias
//   output: pointer to output tensor (N, C_out, L_out)
//   N, C_in, C_out, L_in, L_out, kernel_size, stride, padding, output_padding, groups, dilation
//   stream: CUDA stream
// Note: We assume groups=1 as per contract, but we can support groups>1 if needed.
// We'll implement only the needed case: groups=1, dilation=1.
extern "C" void custom_conv_transpose1d_launcher(const float* input,
                                                const float* weight,
                                                const float* bias,
                                                float* output,
                                                int N,
                                                int C_in,
                                                int C_out,
                                                int L_in,
                                                int L_out,
                                                int kernel_size,
                                                int stride,
                                                int padding,
                                                int output_padding,
                                                int groups,
                                                int dilation,
                                                cudaStream_t stream) {
    // Validate groups and dilation: only support groups=1, dilation=1 for now, as in task.
    if (groups != 1 || dilation != 1) {
        fprintf(stderr, "custom_conv_transpose1d_launcher only supports groups=1, dilation=1\n");
        return;
    }

    cudnnHandle_t handle = get_cudnn_handle();
    cudnnSetStream(handle, stream);

    // Create tensor descriptors.
    // For Conv1d, we use 4D descriptors with height=1.
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t weight_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&weight_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set input descriptor: [N, C_in, 1, L_in]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N, C_in, 1, L_in));

    // Set output descriptor: [N, C_out, 1, L_out]
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N, C_out, 1, L_out));

    // Set filter descriptor: for ConvTranspose, weight shape is [C_in, C_out, kernel_size].
    // In cuDNN filter descriptor for convolution, the format is [C_out, C_in/groups, kernel_h, kernel_w].
    // For ConvTranspose (backward data), the filter dimensions should be [C_in, C_out, 1, kernel_size].
    // However, cuDNN's convolution descriptor uses the filter shape from wDesc, and for backward data it expects the filter as in forward convolution.
    // In PyTorch, for ConvTranspose, the weight is of shape [in_channels, out_channels, kernel_size].
    // cuDNN's cudnnSetFilter4dDescriptor expects format [K, C, R, S] where K = number of output feature maps (i.e., C_out of forward conv), and C = input channels (C_in of forward conv).
    // For backward data, the "dy" is the gradient w.r.t output of forward conv, and "dx" is gradient w.r.t input.
    // So if we want to compute x = conv_transpose(input, weight), we can set:
    //   dy = input (shape [N, C_in, 1, L_in])
    //   dx = output (shape [N, C_out, 1, L_out])
    //   wDesc should describe the forward convolution filter that would produce dy from dx.
    // The forward conv would have input channels = C_out, output channels = C_in, kernel size = kernel_size.
    // So weight_desc should represent shape [C_in, C_out, 1, kernel_size] (K=C_in? Wait: filter dims are [K, C, R, S], where K = number of output feature maps in forward conv. In forward conv, output channels = C_in? That seems swapped.
    // Let's carefully map: In ConvTranspose1d, input has C_in channels, output has C_out channels.
    // PyTorch implementation of ConvTranspose1d uses cudnnConvolutionBackwardData with the weight of shape [in_channels, out_channels, kernel_size].
    // The backward data computes dx given dy and w, where w is the filter of the forward convolution that maps dx -> dy.
    // The forward convolution would be a standard conv with input channels = C_out, output channels = C_in, because dy (gradient w.r.t output) has channels C_in, and dx (gradient w.r.t input) has channels C_out.
    // Therefore, the filter shape for that forward conv is [C_in, C_out, kernel_size] (output channels C_in, input channels C_out).
    // So our weight_desc should be set to [C_in, C_out, 1, kernel_size].
    // This matches PyTorch's weight shape [C_in, C_out, K].
    // We'll set filter as such.
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(weight_desc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           C_in, C_out, 1, kernel_size));

    // Set convolution descriptor for the forward conv (which is being reversed).
    // The forward conv would have stride, padding, dilation as given to ConvTranspose? Actually for ConvTranspose, the parameters stride, padding, dilation refer to the transposed operation, not the forward conv.
    // The forward conv that corresponds to ConvTranspose is a conv with the same stride, padding, dilation, but without output_padding.
    // However, cuDNN's cudnnConvolutionBackwardData computes the gradient of a convolution given the output gradient dy and the filter w.
    // The convolution descriptor passed to cudnnConvolutionBackwardData must describe the forward convolution, i.e., the convolution whose backward data is being computed.
    // In PyTorch, for ConvTranspose with stride s, padding p, dilation d, the forward conv that they use internally has stride s, padding p, dilation d, but not necessarily output_padding.
    // But output_padding is used to increase the output size; it corresponds to additional zero padding on the output of the forward conv (i.e., on dy).
    // Actually, cudnnConvolutionBackwardData has a parameter for the convolution descriptor, and there is no output_padding parameter. PyTorch handles output_padding by modifying the descriptor? Let's check PyTorch source: For ConvTranspose, they call cudnnConvolutionBackwardData with the convolution mode = CUDNN_CROSS_CORRELATION, and they set the padding and stride accordingly, and they handle output_padding by adjusting the input (dy) tensor shape? Actually output_padding adds extra zeros to the output of ConvTranspose, but cuDNN's backward data will compute an output (dx) of size determined by the convolution parameters and input (dy) size.
    // The formula for output size of backward data (dx) given dy size, filter, stride, padding, dilation is: floor((dy_size + 2*padding - dilation*(kernel_size-1) -1)/stride) + 1, but that's the size of the forward conv input for a given output. For backward data, the output size is computed differently: dx_dim = (dy_dim - 1)*stride + dilation*(kernel_size-1) - 2*padding + 1. That is exactly the ConvTranspose output size without output_padding.
    // With output_padding=1, the desired L_out is one larger. To achieve that, PyTorch likely pads the dy tensor (input to ConvTranspose) with extra rows/cols of zeros at the end? Actually output_padding is added to the output size, and since cuDNN doesn't have a direct parameter, PyTorch may allocate a larger dx tensor and then set the conv descriptor with adjusted padding? Let's recall: output_padding < stride. For ConvTranspose, output_padding is equivalent to setting padding = padding - output_padding? No, that would change the relationship.
    // I think PyTorch handles output_padding by using cudnnConvolutionBackwardData with the same conv descriptor as the forward conv that would produce the given dy size, and then if output_padding >0, they explicitly zero out the extra rows (or columns) at the end of the output? Not sure.
    // Safer approach: Instead of relying on cuDNN's backward data directly with output_padding, we can implement ConvTranspose as a standard convolution with fractional stride? That is complex.
    // Alternatively, we can use cudnnConvolutionBackwardData but adjust the input (dy) tensor by adding output_padding zero columns at the end (i.e., increase L_in by output_padding) and then compute expected L_out = (L_in+output_padding -1)*stride -2*padding + dilation*(kernel_size-1)+1. That would produce the correct size? Let's test with numbers: L_in=4096, stride=2, padding=1, dilation=1, kernel=3, output_padding=1. Desired L_out=8192.
    // Without output_padding, formula gives (4096-1)*2 -2 +2 +1 = 8190. So one less.
    // If we increase L_in by 1 (to 4097) and keep same parameters, we get (4097-1)*2 -2 +2 +1 = 8192. So we can simulate output_padding by extending the input (dy) with one zero column at the end (or beginning). That would effectively add an extra stride column, and since output_padding < stride, it works. PyTorch might do this internally.
    // We'll implement this trick: create a temporary input tensor of size [N, C_in, 1, L_in + output_padding] with the last output_padding columns zero, and then call cudnnConvolutionBackwardData with that as dy, and the same conv descriptor (stride, padding, dilation). The output dx will be of size [N, C_out, 1, L_out] as desired.
    // We'll allocate temporary dy on-the-fly using cudaMallocAsync or just cudaMalloc.
    // Complexity: need to copy input and pad with zeros.
    // We'll do this only if output_padding > 0.
    // But for simplicity, since output_padding is passed as 1 in this task, we'll implement the padding.

    // Decide whether to pad.
    int padded_L_in = L_in + output_padding;
    // Create temp input tensor on device if needed.
    float* d_input_padded = nullptr;
    if (output_padding > 0) {
        // Allocate temporary buffer for padded input (N*C_in*1*padded_L_in)
        size_t padded_size = (size_t)N * C_in * 1 * padded_L_in * sizeof(float);
        cudaError_t err = cudaMalloc(&d_input_padded, padded_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "custom_conv_transpose1d_launcher: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            // cleanup and return
            cudnnDestroyTensorDescriptor(input_desc);
            cudnnDestroyTensorDescriptor(output_desc);
            cudnnDestroyFilterDescriptor(weight_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            return;
        }
        // Zero-initialize the padded input.
        cudaMemsetAsync(d_input_padded, 0, padded_size, stream);
        // Copy original input into the beginning of each sample's data.
        // Since input is contiguous [N, C_in, L_in], we need to copy for each batch and channel.
        // We can use cudaMemcpy2DAsync for efficiency: for each batch and channel, copy L_in floats from input to padded with row pitch padded_L_in.
        // But input is contiguous with row pitch L_in. We'll launch a small kernel or use cudaMemcpy2D.
        // Better: use cudaMemcpy2DAsync per batch/channel, but that's many calls. For N*C_in = 128*3=384 calls, okay.
        // Alternatively, use a simple CUDA kernel to copy and pad, but we can use cudaMemcpy2DAsync.
        // We'll loop over N and C_in.
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C_in; c++) {
                const float* src = input + (n * C_in + c) * L_in;
                float* dst = d_input_padded + (n * C_in + c) * padded_L_in;
                cudaMemcpy2DAsync(dst, padded_L_in * sizeof(float),
                                  src, L_in * sizeof(float),
                                  L_in * sizeof(float), 1,
                                  cudaMemcpyDeviceToDevice, stream);
            }
        }
        // Now use d_input_padded as the dy tensor.
        input = d_input_padded; // constant pointer, but we'll create a new descriptor with padded_L_in.
    }

    // Update input descriptor if padded.
    if (output_padding > 0) {
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               N, C_in, 1, padded_L_in));
    }

    // Set convolution descriptor: forward conv parameters.
    // For ConvTranspose, the forward conv (which we are taking backward data of) typically has:
    // pad = padding, stride = stride, dilation = dilation.
    // However, there is a subtlety: The forward conv would need to produce dy of size (padded_L_in) from an input of size L_out, i.e., the forward conv input size = L_out, output size = padded_L_in.
    // The relation must hold: padded_L_in = (L_out - 1)*stride - 2*padding + dilation*(kernel_size-1) + 1.
    // This is consistent with our earlier formula.
    // So we set conv descriptor with those parameters.
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                padding, 0, // vertical padding, horizontal padding
                                                stride, 1,   // vertical stride, horizontal stride
                                                dilation, 1, // vertical dilation, horizontal dilation
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    // Note: The convolution descriptor is 2D, but we are using 4D tensors with height=1, so vertical parameters are irrelevant.

    // Set up algorithm selection.
    cudnnConvolutionBwdDataAlgo_t algo;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        handle,
        weight_desc,
        input_desc, // dy descriptor (the gradient w.r.t output of forward conv)
        conv_desc,
        output_desc, // dx descriptor (the gradient w.r.t input of forward conv)
        CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
        0,
        &algo));

    // Determine workspace size.
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
        handle,
        weight_desc,
        input_desc,
        conv_desc,
        output_desc,
        algo,
        &workspace_size));

    // Allocate workspace if needed.
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaError_t err = cudaMalloc(&workspace, workspace_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "custom_conv_transpose1d_launcher: cudaMalloc workspace failed: %s\n", cudaGetErrorString(err));
            // cleanup and return
            if (d_input_padded) cudaFree(d_input_padded);
            cudnnDestroyTensorDescriptor(input_desc);
            cudnnDestroyTensorDescriptor(output_desc);
            cudnnDestroyFilterDescriptor(weight_desc);
            cudnnDestroyConvolutionDescriptor(conv_desc);
            return;
        }
    }

    // Perform the backward data convolution (transposed conv).
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionBackwardData(
        handle,
        &alpha,
        weight_desc, weight,
        input_desc, input, // dy
        conv_desc,
        algo,
        workspace, workspace_size,
        &beta,
        output_desc, output)); // dx

    // Add bias if provided.
    if (bias != nullptr) {
        // Add bias to each output channel: output[n, c, l] += bias[c]
        // We can use cudnnAddTensor or just a simple kernel. Since output is contiguous, we can launch a simple kernel.
        // We'll define a small kernel below or use cudnnAddTensor.
        // We'll use cudnnAddTensor for simplicity.
        cudnnTensorDescriptor_t bias_desc;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&bias_desc));
        // Bias descriptor: shape [1, C_out, 1, 1]
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(bias_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               1, C_out, 1, 1));
        float alpha_bias = 1.0f;
        float beta_bias = 1.0f;
        CHECK_CUDNN(cudnnAddTensor(handle, &alpha_bias, bias_desc, bias, &beta_bias, output_desc, output));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(bias_desc));
    }

    // Cleanup.
    if (workspace) cudaFree(workspace);
    if (d_input_padded) cudaFree(d_input_padded);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(weight_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);

    // Note: no explicit synchronize; caller will synchronize via stream.
    // However, we should check for errors if needed.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "custom_conv_transpose1d_launcher: CUDA error after launch: %s\n", cudaGetErrorString(err));
    }
}
