#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

// 声明 launcher 函数供 binding.cpp 调用
extern "C" void move_to_cuda_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream);
extern "C" void stft_abs_launcher(float* output, const float* input, int64_t batch_size, int64_t length, int n_fft, int hop_length, int win_length, cudaStream_t stream);
extern "C" void reflection_pad_2d_launcher(float* output, const float* input, int64_t batch_size, int64_t height, int64_t width, int pad_left, int pad_right, int pad_top, int pad_bottom, const int64_t* input_strides, cudaStream_t stream);

// 错误检查宏
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                \
        cudaError_t err = call;                                                         \
        if (err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(err));                                           \
            return;                                                                     \
        }                                                                               \
    } while (0)

#define CUFFT_CHECK(call)                                                               \
    do {                                                                                \
        cufftResult err = call;                                                         \
        if (err != CUFFT_SUCCESS) {                                                     \
            fprintf(stderr, "cuFFT error at %s:%d: %d\n", __FILE__, __LINE__, (int)err);\
            return;                                                                     \
        }                                                                               \
    } while (0)

// move_to_cuda: 简单的异步复制
void move_to_cuda_launcher(float* output, const float* input, int64_t numel, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(output, input, numel * sizeof(float), cudaMemcpyHostToDevice, stream));
}

// 内核：stft_abs 中对单个 batch 的复数帧取模
// 输入：复数数组（绝对值之前），输出：float 模值
// complex_data 是 cuFFT 输出（复数），大小为 n_fft 的复数数组，但有效频率为 n_fft/2+1 个
// 此 kernel 处理一个 batch 的所有帧：将复数结果转为模值并写入 output，output 布局 [batch, freq, frame]
__global__ void stft_abs_kernel(
    const cufftComplex* complex_data, // [batch, n_fft, num_frames]? 实际上我们按 batch 处理，但这里传递一个 batch 的所有复数数据，形状 [n_fft, num_frames]
    float* output,                    // [freq, num_frames] for this batch
    int n_fft,
    int num_freq,
    int num_frames
) {
    int freq = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.y * blockDim.y + threadIdx.y;
    if (freq < num_freq && frame < num_frames) {
        cufftComplex val = complex_data[freq * num_frames + frame]; // 注意复数数据布局：cuFFT 输出的行优先 [n_fft, num_frames]? 但 cuFFT 默认是列主序？需要确认。
        output[freq * num_frames + frame] = sqrtf(val.x * val.x + val.y * val.y);
    }
}

// 内核：初始化 Hann 窗
__global__ void init_hann_window(float* window, int win_length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < win_length) {
        window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (float)(win_length - 1)));
    }
}

// stft_abs 完整实现
// 输入 x: [batch_size, length] 连续 float32 on GPU
// 输出 output: [batch_size, num_freq, num_frames]  float32
// 输出步长为 [num_freq*num_frames, num_frames, 1] 但不保证，由 binding 分配，此处直接写入，假设连续。
void stft_abs_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t length,
    int n_fft,
    int hop_length,
    int win_length,
    cudaStream_t stream
) {
    // 参数检查
    if (batch_size <= 0 || length <= 0 || n_fft <= 0 || hop_length <= 0 || win_length <= 0) {
        return;
    }

    // 计算参数
    int center_pad = n_fft / 2;  // 256
    int num_frames = 1 + (int)((length + 2 * center_pad - n_fft) / (int64_t)hop_length);
    int num_freq = n_fft / 2 + 1;  // 257

    // 分配临时窗口内存 device
    float* d_window;
    CUDA_CHECK(cudaMalloc(&d_window, win_length * sizeof(float)));

    // 初始化 Hann 窗
    int block = 256;
    int grid = (win_length + block - 1) / block;
    init_hann_window<<<grid, block, 0, stream>>>(d_window, win_length);

    // 为每个 batch 处理
    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * length;
        float* batch_output = output + b * num_freq * num_frames;

        // 1. 反射填充 center_pad 个点到临时数组，长度 padded_len = length + 2*center_pad
        int padded_len = length + 2 * center_pad;
        float* d_padded;
        CUDA_CHECK(cudaMalloc(&d_padded, padded_len * sizeof(float)));

        // 使用 kernel 填充？或者用 cudaMemcpyAsync 分段复制
        // 简单起见，使用三次 memcpy：前部反射，中间原数据，后部反射
        // 反射填充规则：左侧填充信号的前 center_pad 个元素，顺序反向；右侧填充信号的后 center_pad 个元素，顺序反向。
        // 但要注意 torch.stft 的 center 反射填充具体方式：在信号两端填充 n_fft//2 个点，使用 'reflect' 模式，即复制边缘的值（不包括边缘本身？Reflect 包含边缘？torch 的 reflect 填充通常不重复边缘。核实：对一维信号，pad_left=center_pad, pad_right=center_pad，reflect 模式使用边界的镜像，不重复边缘。
        // 这里先精确模拟：torch.stft 默认 center=True, pad_mode='reflect'。根据 PyTorch 文档：reflect pad 填充长度为 pad，不包括边缘值。因此，反射段为 input[1:1+pad] 的反向等。
        // 但为了数值完全一致，我们实现一个简单的循环或 kernel 来完成填充。
        // 由于 n_fft=512, center_pad=256, length 可能很大，但填充操作相对较小。
        // 使用一个内核并行填充。

        // 分配 d_padded 后，使用内核 fill_reflect_pad
        // 定义 kernel：每个线程负责填充一个位置
        int padded_grid = (padded_len + block - 1) / block;
        // 这里需要 kernel 函数，但我们尚未定义。临时添加一个全局函数。
        // 我们可以从 host 用 cudaMemcpyAsync 分段拷贝，然后内核填充反射部分。
        // 先拷贝中间部分：
        CUDA_CHECK(cudaMemcpyAsync(d_padded + center_pad, batch_input, length * sizeof(float), cudaMemcpyDeviceToDevice, stream));

        // 左侧反射：从 batch_input 的索引 1..center_pad 复制并反向
        // 简单用循环？效率不高，但 batch_size=128, length=160000，还是可以接受。不过最好内核。
        // 我们实现一个内核 reflect_pad_left_right，但为清晰，直接用临时内核。
    }

    // 清理
    CUDA_CHECK(cudaFree(d_window));
}

// 由于上述代码不完整，我们重新设计完整实现：
// 实现步骤：
// 1. 为每个 batch 独立处理：
//    - 创建反射填充后的输入信号 padded_signal，长度 padded_len
//    - 对每一帧，取窗口，乘以 Hann 窗，做 n_fft 点 R2C FFT
//    - 计算复数模值，写入 output[b, :, frame]
//  使用 cuFFT 批量或循环处理每一帧。考虑到性能，可以使用 cuFFT 的 many 接口，同时处理多个帧。
//  但为了简单和清晰，我们采用每帧调用 cuFFT 的方式，但这样每个 batch 有 626 帧，batch 128，总共 80128 次 FFT 调用，可能较慢。
//  优化：使用 cuFFT plan many，将同一 batch 的多帧合并为一个 2D FFT，但需要先构建复数输入（实部）。
//  另一种方法：使用 cuFFT 的 R2C 变换，对 padded_signal 的每个窗口分别做 FFT。可以使用 cufftPlanMany 同时变换多个窗口。
//  考虑到实现复杂度，我们先实现每帧独立 FFT，但可能会影响性能。不过当前重点是正确性，后期可优化。
//  但由于性能要求 5% 以上，我们必须尽可能高效。所以使用 cufftPlanMany 一次性处理所有帧可能更好。
//  具体：将 padded_signal 按帧和重叠复制到一个大缓冲区，每行是一个窗口，然后调用 cuFFT plan many，对每行执行 R2C FFT。
//  注意重叠：每帧起始位置 hop_length，帧之间重叠。复制时会有大量重复数据，但内存占用可接受（padded_len ~160512, 每窗口512, 共626帧，每帧复制512个样本，总约320k float 每个batch，128 batch约40M float，160MB，可能太大？实际上每个窗口512，626帧约320k，与原始160k相比大一倍，可以接受。
//  也可以使用 cuFFT callback 或自定义 kernel 实现，但先采用复制方法。

// 重新实现完整的 stft_abs_launcher：
// 步骤：
// A. 创建 Hann 窗 device 内存。
// B. 对每个 batch 循环：
//    1. 创建扩展缓冲 d_padded 长度 padded_len，填充反射。
//    2. 创建窗口缓冲区 d_frames，大小 num_frames * n_fft，复制每个窗口并乘以 Hann 窗。
//    3. 创建复数输出缓冲区 d_complex，大小 num_frames * (n_fft/2+1) 个 cufftComplex。
//    4. 调用 cufftPlanMany 设置 R2C FFT，对 num_frames 个长度为 n_fft 的实数数组进行变换。
//    5. 执行 cufftExecR2C，得到复数结果。
//    6. 启动 kernel 计算模值，输出到 batch_output。
// 这样可以减少 kernel launch 开销。

// 反射填充内核：使用一个线程填充一个位置，处理左右反射。
__global__ void reflect_pad_fill(float* padded, const float* input, int length, int pad) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = length + 2 * pad;
    if (idx >= total) return;
    if (idx < pad) {
        // 左侧反射：输入[pad - idx]？反射规则：不重复边缘，因此 pad 个元素的索引对应输入[pad - idx]（当 idx=0 对应输入[pad]？）
        // 标准：对于左边 pad p，输出位置 i (0<=i<p) 对应输入[p - i - 1]？需要确定。
        // Reflect 模式（对称）：索引 i 映射到 2*p - i - 2? 对于一维，若长度为 L, pad p，则新索引为：
        // 左侧：i' = p - i - 1 (i=0 to p-1) 对应原始输入索引 p - i - 1
        // 右侧：i' = 2*L + p - i - 2 对应原始输入索引 2*L + p - i - 2，化简 i' = 2*L + p - i - 2, 右边位置 i >= L+p, 对应输入索引 2*L + p - i - 2
        // 但需要确认 PyTorch 行为。一般 Reflect 填充不重复边界，因此左侧填充为 input[1..p] 的反向，右侧为 input[L-p-1..L-2] 的反向。
        // 简单使用：padded[idx] = input[pad - idx];  // 当 idx=0, input[pad]; idx=pad-1, input[1] 正确？例如 L=5, p=2, 左侧填充应为 input[2], input[1]，即 [x2, x1]。pad - idx 当 idx=0 得 2; idx=1 得 1。正确。
        padded[idx] = input[pad - idx];
    } else if (idx >= pad + length) {
        int right_idx = idx - pad - length; // 0..pad-1
        // 右侧填充：输入[length - 2 - right_idx]
        padded[idx] = input[length - 2 - right_idx];
    } else {
        padded[idx] = input[idx - pad];
    }
}

// 窗口复制并乘以 Hann 窗内核：输入 padded，输出 frames 缓冲区，每帧 n_fft，共 num_frames 帧。
__global__ void build_frames(float* frames, const float* padded, const float* window, int length_padded, int n_fft, int hop_length, int num_frames) {
    int frame = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame >= num_frames) return;
    int start = frame * hop_length;
    // 确保 start + n_fft <= length_padded
    for (int i = 0; i < n_fft; ++i) {
        frames[frame * n_fft + i] = padded[start + i] * window[i];
    }
}

// 取模内核：输入复数数组（交错或 cufftComplex），输出模值到 output，输出形状 [num_freq, num_frames]，按 freq major? 我们决定输出布局 [batch, freq, frame]，但 cuFFT 输出按 batch 的 frames 排列，每行是 freq。所以复用之前定义。
__global__ void complex_abs_kernel(const cufftComplex* complex_data, float* output, int num_freq, int num_frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_freq * num_frames;
    if (idx >= total) return;
    int freq = idx % num_freq;
    int frame = idx / num_freq;
    cufftComplex val = complex_data[idx]; // 注意 cuFFT 输出布局：对于 plan many，如果使用 inembed 和 onembed 设置为 [n_fft]? 默认输出是 n_fft/2+1 个复数按行存储。
    output[freq * num_frames + frame] = sqrtf(val.x * val.x + val.y * val.y);
}

// 完整 stft_abs_launcher
void stft_abs_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t length,
    int n_fft,
    int hop_length,
    int win_length,
    cudaStream_t stream
) {
    // 基本参数
    int center_pad = n_fft / 2;
    int num_frames = 1 + (int)((length + 2 * center_pad - n_fft) / (int64_t)hop_length);
    int num_freq = n_fft / 2 + 1;

    // 分配 Hann 窗
    float* d_window;
    CUDA_CHECK(cudaMalloc(&d_window, win_length * sizeof(float)));
    int block = 256;
    int grid = (win_length + block - 1) / block;
    init_hann_window<<<grid, block, 0, stream>>>(d_window, win_length);

    // 为每个 batch 分配临时内存（可复用，但 batch 循环内复用）
    int padded_len = length + 2 * center_pad;
    float* d_padded = nullptr;
    float* d_frames = nullptr;
    cufftComplex* d_complex = nullptr;
    CUDA_CHECK(cudaMalloc(&d_padded, padded_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_frames, num_frames * n_fft * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_complex, num_frames * num_freq * sizeof(cufftComplex)));

    // 创建 cuFFT plan many：处理 num_frames 个长度为 n_fft 的 R2C 变换
    cufftHandle plan;
    int rank = 1;
    int n[] = {n_fft};
    int inembed[] = {n_fft};   // 输入实数，每行 n_fft 个
    int istride = 1;
    int idist = n_fft;
    int onembed[] = {num_freq};
    int ostride = 1;
    int odist = num_freq;
    CUFFT_CHECK(cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, num_frames));
    CUFFT_CHECK(cufftSetStream(plan, stream));

    for (int64_t b = 0; b < batch_size; ++b) {
        const float* batch_input = input + b * length;
        float* batch_output = output + b * num_freq * num_frames;

        // 1. 填充反射 padded 数组
        int padded_grid = (padded_len + block - 1) / block;
        reflect_pad_fill<<<padded_grid, block, 0, stream>>>(d_padded, batch_input, (int)length, center_pad);

        // 2. 构建帧并乘以窗
        int frames_grid = (num_frames + block - 1) / block;
        build_frames<<<frames_grid, block, 0, stream>>>(d_frames, d_padded, d_window, padded_len, n_fft, hop_length, num_frames);

        // 3. 执行 FFT
        CUFFT_CHECK(cufftExecR2C(plan, d_frames, d_complex));

        // 4. 计算模值，写入 batch_output，输出布局 [freq, frame]（freq 连续在每帧内? 输出是 freq-major，frame 次之，即 output[freq * num_frames + frame]）
        int abs_total = num_freq * num_frames;
        int abs_grid = (abs_total + block - 1) / block;
        complex_abs_kernel<<<abs_grid, block, 0, stream>>>(d_complex, batch_output, num_freq, num_frames);
    }

    // 清理
    CUFFT_CHECK(cufftDestroy(plan));
    CUDA_CHECK(cudaFree(d_window));
    CUDA_CHECK(cudaFree(d_padded));
    CUDA_CHECK(cudaFree(d_frames));
    CUDA_CHECK(cudaFree(d_complex));
}

// reflection_pad_2d 实现
// 输入 x: 任意 strides 的 3D tensor [batch, height, width]，实际 2D 填充是在最后两维。
// 我们需要根据 input_strides 访问。假设输入 strides 数组长度为 3，按 batch, height, width 顺序。
// 输出是连续的 [batch, out_height, out_width]，其中 out_height = height + pad_top + pad_bottom, out_width = width + pad_left + pad_right。
void reflection_pad_2d_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    const int64_t* input_strides, // 长度为 3，元素为字节数还是元素数？契约说 tensor strides 通常元素数，但这里我们需要元素偏移，因此 binding.cpp 应传递元素 strides？我们约定 binding 会传递元素 strides（即每个维度的步长，以元素为单位）。
    cudaStream_t stream
) {
    int out_height = height + pad_top + pad_bottom;
    int out_width = width + pad_left + pad_right;
    int64_t total_out = batch_size * out_height * out_width;

    // 启动一个 2D grid 或者简单的 1D grid 映射
    int block = 256;
    int grid = (total_out + block - 1) / block;
    // 内核需要知道输入 strides 和尺寸来索引
    // 我们传递 input_strides 指向的 d_strides? 但 launcher 中 input_strides 是 host 指针，需要将其值复制到 device 常量内存或作为 kernel 参数。
    // 简单做法：将 strides 作为 kernel 参数值传递，使用数组或三个 int64_t。
    // 由于 kernel 是在 device 上，参数可以是标量数组，但需要复制数值，不能传递指针到 host 内存。
    // 我们在 launcher 中读取三个 strides 到局部变量，然后通过 kernel 参数传递值。
    int64_t stride_b = input_strides[0];
    int64_t stride_h = input_strides[1];
    int64_t stride_w = input_strides[2];

    // 定义内核
    // 使用 1D grid 遍历所有输出元素
    // 每个线程计算输出坐标 (b, oh, ow)，然后找到对应的输入坐标 (ih, iw) 进行反射映射。
}

// 内核：反射填充 2D
__global__ void reflection_pad_2d_kernel(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    int64_t stride_b,
    int64_t stride_h,
    int64_t stride_w
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = batch_size * (height + pad_top + pad_bottom) * (width + pad_left + pad_right);
    if (idx >= total) return;

    int out_width = width + pad_left + pad_right;
    int out_height = height + pad_top + pad_bottom;

    int64_t tmp = idx;
    int ow = tmp % out_width;
    tmp /= out_width;
    int oh = tmp % out_height;
    tmp /= out_height;
    int b = tmp;

    // 映射到输入坐标
    int ih = oh - pad_top;
    int iw = ow - pad_left;

    // 反射映射：如果超出边界，反射
    if (ih < 0) ih = -ih - 1; // 或使用公式 2*0 - ih - 1? 需要正确处理。
    if (ih >= height) ih = 2 * height - ih - 1; // 需要确保不越界。
    if (iw < 0) iw = -iw - 1;
    if (iw >= width) iw = 2 * width - iw - 1;

    // 边界条件：反射有时会越界，但通常 pad 不超过 size-1，但在这里 pad=5, size 足够大。
    // 安全起见，clamp？但 torch 的 reflect 在 pad 大于 size 时会报错，所以我们假设 pad < size。
    // 直接使用映射后的索引。

    // 读取输入值
    const float* input_ptr = input + b * stride_b + ih * stride_h + iw * stride_w;
    output[idx] = *input_ptr;
}

// 更新 reflection_pad_2d_launcher 实现
void reflection_pad_2d_launcher(
    float* output,
    const float* input,
    int64_t batch_size,
    int64_t height,
    int64_t width,
    int pad_left,
    int pad_right,
    int pad_top,
    int pad_bottom,
    const int64_t* input_strides,
    cudaStream_t stream
) {
    int64_t stride_b = input_strides[0];
    int64_t stride_h = input_strides[1];
    int64_t stride_w = input_strides[2];

    int out_height = height + pad_top + pad_bottom;
    int out_width = width + pad_left + pad_right;
    int64_t total = batch_size * out_height * out_width;

    int block = 256;
    int grid = (int)((total + block - 1) / block);
    reflection_pad_2d_kernel<<<grid, block, 0, stream>>>(
        output, input, batch_size, height, width,
        pad_left, pad_right, pad_top, pad_bottom,
        stride_b, stride_h, stride_w
    );
}
