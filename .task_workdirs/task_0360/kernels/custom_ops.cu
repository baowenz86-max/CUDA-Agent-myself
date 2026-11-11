// CUDA kernels for RNN forward
#include <cuda_runtime.h>

// Kernel: 执行两个时间步的 RNN 计算
// 每个 block 处理一个时间步和 batch 中的所有样本
// 使用共享内存缓存权重和中间结果
template <int input_size, int hidden_size>
__global__ void rnn_forward_kernel(
    const float* __restrict__ input,       // [seq_length, batch, input_size]
    const float* __restrict__ weight_ih_l0, // [hidden_size, input_size]
    const float* __restrict__ weight_hh_l0, // [hidden_size, hidden_size]
    const float* __restrict__ bias_ih_l0,   // [hidden_size]
    const float* __restrict__ bias_hh_l0,   // [hidden_size]
    const float* __restrict__ weight_ih_l1, // [hidden_size, hidden_size]
    const float* __restrict__ weight_hh_l1, // [hidden_size, hidden_size]
    const float* __restrict__ bias_ih_l1,   // [hidden_size]
    const float* __restrict__ bias_hh_l1,   // [hidden_size]
    const float* __restrict__ hx,          // optional [num_layers, batch, hidden_size] or nullptr
    float* __restrict__ output,            // [seq_length, batch, hidden_size]
    float* __restrict__ h_n,               // [num_layers, batch, hidden_size]
    int seq_length,
    int batch_size,
    int num_layers
) {
    // 共享内存布局：
    // 第一层权重 (hidden_size * input_size) + 第二层权重 (hidden_size * hidden_size * 2) 
    // 中间变量：h_prev_l0 (batch * hidden_size), h_cur_l0, h_prev_l1, h_cur_l1
    extern __shared__ float smem[];
    float* smem_weight_ih_l0 = smem; // hidden_size * input_size
    float* smem_weight_hh_l0 = smem_weight_ih_l0 + hidden_size * input_size; // hidden_size * hidden_size
    float* smem_weight_ih_l1 = smem_weight_hh_l0 + hidden_size * hidden_size; // hidden_size * hidden_size
    float* smem_weight_hh_l1 = smem_weight_ih_l1 + hidden_size * hidden_size; // hidden_size * hidden_size
    float* smem_h_prev_l0 = smem_weight_hh_l1 + hidden_size * hidden_size; // batch * hidden_size
    float* smem_h_cur_l0 = smem_h_prev_l0 + batch_size * hidden_size; // batch * hidden_size
    float* smem_h_prev_l1 = smem_h_cur_l0 + batch_size * hidden_size; // batch * hidden_size
    float* smem_h_cur_l1 = smem_h_prev_l1 + batch_size * hidden_size; // batch * hidden_size

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int batch_idx = blockIdx.x; // 一个 block 处理一个 batch 样本

    // 将权重加载到共享内存 (仅 block 0 需要，但每个 block 都加载，因为权重相同)
    // 为了简单，每个 block 都加载权重到各自的共享内存，虽然浪费但简化同步
    // 实际可以通过第一个 block 加载然后 __syncthreads 共享，但不同 block 共享内存不共享，所以每个 block 必须加载自己的拷贝
    // 权重大小：64*100 + 64*64 + 64*64 + 64*64 = 6400 + 4096 + 4096 + 4096 = 18688 floats = ~75KB
    // 加上中间变量 16*64*4 = 4096 floats，总共约 91KB，可能超过 48KB 默认限制，需要调整 shared memory size 或分段加载
    // 这里我们选择不分段加载权重，而是直接计算时从 global 读取，但在 block 内通过循环利用 cache。实际上，我们可以不将权重放入共享内存，而是依赖 L1/L2 cache。
    // 然而，为了性能，我们可以使用较小的 shared memory 来缓存部分权重。
    // 但为简化，这里不使用共享内存缓存权重，而是直接从 global 读取。
    // 不过，我们可以利用共享内存缓存中间隐藏状态，因为每个时间步需要多次访问。
    // 这个 kernel 的设计是每个时间步迭代整个 seq_length，但 thread 分工为：每个 thread 计算一个 hidden unit 的输出。
    // 由于 hidden_size=64，我们可以用 64 个线程，但 block size 可以更大来覆盖 batch。
    // 更好的设计：每个 block 处理一个时间步的全部 batch，block 内 thread 分别负责计算每个 batch 样本的每个 hidden unit。
    // 即 gridDim.x = seq_length, blockDim.x = batch_size * hidden_size? 但 batch_size=16, hidden_size=64 => 1024 threads，可行。
    // 然后每个 thread 串行计算一个 batch 中一个 hidden unit 的输出。
    // 但这样每个 thread 需要访问所有权重和输入，计算量大。
    // 我们设计更有效的 kernel：
    // 每个 block 处理一个 batch 样本的所有 hidden units 和所有时间步？这样需要长时间循环，且难以并行。
    // 推荐：每个 block 处理一个时间步和一个 batch 样本，block 内 threads 并行计算 hidden_size 个输出。
    // 这样 grid = (seq_length, batch_size)，block = (hidden_size) = 64 threads。
    // 对于每一层，每个 thread 计算一个 hidden unit：
    //   h_i = tanh( sum_j x_j * W_ih[i][j] + b_ih[i] + sum_k h_prev_k * W_hh[i][k] + b_hh[i] )
    // 这需要读取输入向量 x (input_size=100) 和 h_prev (hidden_size=64)。
    // 使用线程并行计算点积，需要减少。可以用 warp reduction，但 hidden_size=64 正好一个 warp (32) 不够，两个 warp？
    // 或者 block 内所有线程协作计算每个输出。更好的：block 大小设为 64，每个线程计算一个输出，但每个线程内部串行对 100+64 项求和。
    // 这种简单方法对于 input_size 100 和 hidden_size 64 可能性能不错。
    // 我们需要考虑两层：第一层的输入是 x，第二层的输入是第一层的输出。因此，我们需要顺序计算。
    // 由于时间步依赖，我们需要按时间步迭代，每个时间步内，先计算第一层所有 batch 样本的输出，然后计算第二层。
    // 我们可以用两个 kernel：一个计算一层，但这里用单个 kernel 处理两层，以避免中间存储。
    // 但为了简单和正确性，我们可以在同一个 kernel 中顺序处理每个时间步，使用 grid 大小为 batch_size，block 大小为 hidden_size，但这样时间步需要外部循环（在 kernel 内循环 seq_length 次）。
    // 然而，时间步依赖使得每个时间步必须等待前一时间步完成，这会降低并行性。但我们可以利用 grid-stride 方式对 batch 并行。
    // 考虑性能，最直接的办法是用 cuBLAS 做矩阵乘法，但任务要求非卷积 CUDA kernel，所以这里手写。
    // 最终设计：一个 kernel 处理一个 batch 样本（blockIdx.x），block 内 64 线程，每个线程负责计算一个 hidden unit 的输出，循环所有时间步。
    // 这样 grid = batch_size, block = hidden_size (64)。每个 block 独立处理该 batch 样本，共享内存可以缓存该 batch 的 h_prev 和 x_t。
    // 但这样每个 block 需要串行 seq_length=2048 步，可能太慢。并行度只有 batch=16，太少。
    // 为了更好的并行，我们可以在时间维度上并行，但要处理依赖。一种方法是用扫描算法，但复杂。
    // 鉴于本任务规模，我们先用简单版本：grid = (seq_length, batch)，每个 block 处理一个 (t, b) 对，block 内计算该时间步该样本的两个层。
    // 但这样每个 block 需要前一时刻的 h_prev，这需要从 global 读取，并且需要确保前一时刻已经计算完成。我们不能保证顺序，因为所有 blocks 可能同时执行，需要依赖链。
    // 因此，我们必须按时间步顺序启动 kernel 或使用屏障。但单一 kernel 无法跨 block 同步（除非使用 cooperative groups）。
    // 所以正确的方法是：在 launcher 中对时间步循环，为每个时间步启动一个 kernel，或者使用一个 kernel 按时间步迭代，但只使用一个 block 处理所有 batch。
    // 一个 block 最多 1024 线程，可以处理 batch=16 * hidden=64 = 1024 个输出单元，正好！这样我们可以用一个 block 处理所有 batch 的所有 hidden units，依次计算每个时间步。
    // block 内线程分工：threadIdx.x 对应 batch * hidden + hidden_unit。这样每个线程负责计算一个 batch 样本的一个 hidden unit。
    // 然后在时间步循环中，每个线程先计算第一层输出，存储在共享内存中，然后同步，再计算第二层输出，存储到 global output，并更新 h_prev。
    // 共享内存需要存储第一层输出 h1_smem[batch][hidden] 和更新后的第二层 h_prev。
    // 这个设计简单且正确，但只使用一个 block，GPU 利用率低，但 seq_length=2048, 每个时间步 1024 线程计算，可能性能还可以。
    // 然而，我们可以让每个 block 处理一部分 batch，例如 grid = batch_size，block = hidden_size，这样每个 block 处理一个 batch 样本，但时间步循环在 block 内部。每个 block 独立处理一个 batch，它们之间不需要通信。
    // 这样并行度是 batch=16，每个 block 有 64 线程，循环 2048 步，每步每个线程计算一个 hidden unit 的两个层。
    // 共享内存可缓存该 batch 的 h_prev。
    // 这个版本可能更好，因为块更多（16 个），但每个块 64 线程，能占用多个 SM。
    // 我们选择此设计：grid = batch_size, block = hidden_size。
    // 但需要注意：如果 batch_size > SM 数量，可以同时运行。
    // 线程映射：threadIdx.x = hidden_unit (0..hidden_size-1)。
    // 对于每个时间步 t（循环内部，从 0 到 seq_length-1）：
    //   1. 每个线程计算第一层的 pre-activation： sum_{j=0}^{input_size-1} input[t, batch, j] * weight_ih_l0[hidden_unit, j] + bias_ih_l0[hidden_unit]
    //       + sum_{k=0}^{hidden_size-1} h_prev_l0[k] * weight_hh_l0[hidden_unit, k] + bias_hh_l0[hidden_unit]
    //      这个需要读取 input[t, batch, :] 和 h_prev_l0[:]。
    //   2. 应用 tanh，存储到共享内存 h1_smem[hidden_unit] (每个 batch 私有)。
    //   3. __syncthreads() 确保所有线程完成第一层，并且 h1_smem 已写入。
    //   4. 每个线程计算第二层的 pre-activation： sum_{j=0}^{hidden_size-1} h1_smem[j] * weight_ih_l1[hidden_unit, j] + bias_ih_l1[hidden_unit]
    //       + sum_{k=0}^{hidden_size-1} h_prev_l1[k] * weight_hh_l1[hidden_unit, k] + bias_hh_l1[hidden_unit]
    //   5. 应用 tanh，将结果写入 output[t, batch, hidden_unit]，并更新 h_prev_l1[hidden_unit]。
    //   6. 更新 h_prev_l0[hidden_unit] = h1_smem[hidden_unit]。
    //   7. __syncthreads() 确保 h_prev 更新完成（虽然下一时间步读取，需要同步）。
    // 最后，将 h_prev_l0 和 h_prev_l1 写入 h_n[0, batch, :] 和 h_n[1, batch, :]。
    // 如果 num_layers != 2，需要调整，但契约固定为 2。
    // 共享内存需要 h1_smem[hidden_size] 和 h_prev_l0[hidden_size], h_prev_l1[hidden_size]。
    // 也可以使用寄存器来存储每个线程自己的 h_prev 值？因为每个线程只负责一个 hidden unit，它只需要知道其他 hidden units 的 h_prev 值用于第二层的输入（h1_smem）和第一层？不，第一层需要 h_prev_l0 的所有值，不仅自己的。所以必须共享。
    // 因此，共享内存包括 h_prev_l0 和 h1_smem。h_prev_l1 也需要共享，因为每个线程需要读取其他线程的 h_prev_l1 用于 second layer recurrent。
    // 所以共享内存：float h_prev_l0[hidden_size], float h1_smem[hidden_size], float h_prev_l1[hidden_size]，都是 block 私有。
    // hidden_size=64，所以 3*64 floats = 768 bytes，很小。
    
    // 注意：以下实现假设 batch_size 和 hidden_size 是编译时常量，但为了灵活性，使用模板参数。
    // 然而，模板参数必须在编译时确定，但实际值是 64 和 100。我们可以为这些尺寸特化。
    // 任务固定 input_size=100, hidden_size=64，所以我们在 launcher 中针对这些尺寸调用模板 kernel。
    // 也可以使用动态维度，但性能较低。
    
    // 由于我们只有模板 kernel 定义，这里给出通用代码，假设 input_size 和 hidden_size 作为模板参数。
    // 在 launcher 中根据实际尺寸实例化。
    // 下面的函数体是模板，但需要注意边界检查。
    // 我们还需要在 kernel 内部处理 hx 可选。
    
    // 以下是通用逻辑，但使用模板参数避免循环边界判断。
    // 由于使用了模板参数，循环边界固定，编译器可以优化。
    
    // 但为了简洁，我们这里使用编译时常量，并在 launcher 中 for 循环动态调用？不行，模板参数必须编译时确定。
    // 我们将提供两个特化：input_size=100, hidden_size=64；以及潜在的其他组合？但契约只要求这些。
    // 如果尺寸变化，可以添加其他实例。
    // 我们将在 launcher 中检查尺寸，如果匹配则调用专门的 kernel；否则报错或回退到通用 kernel。
    // 由于任务固定尺寸，我们只实现一个特化。
    
    // 因此，我们定义非模板 kernel，硬编码 hidden_size=64, input_size=100。
    // 这样更简单。
}

// 为非模板，使用固定尺寸
__global__ void rnn_forward_fixed(
    const float* __restrict__ input,       // [seq_length, batch, 100]
    const float* __restrict__ weight_ih_l0, // [64, 100]
    const float* __restrict__ weight_hh_l0, // [64, 64]
    const float* __restrict__ bias_ih_l0,   // [64]
    const float* __restrict__ bias_hh_l0,   // [64]
    const float* __restrict__ weight_ih_l1, // [64, 64]
    const float* __restrict__ weight_hh_l1, // [64, 64]
    const float* __restrict__ bias_ih_l1,   // [64]
    const float* __restrict__ bias_hh_l1,   // [64]
    const float* __restrict__ hx,          // [2, batch, 64] or nullptr
    float* __restrict__ output,            // [seq_length, batch, 64]
    float* __restrict__ h_n,               // [2, batch, 64]
    int seq_length,
    int batch_size
) {
    const int hidden_size = 64;
    const int input_size = 100;
    const int num_layers = 2;

    int batch_idx = blockIdx.x;
    int hidden_unit = threadIdx.x;

    if (batch_idx >= batch_size || hidden_unit >= hidden_size) return;

    // 共享内存：h_prev_l0, h1_smem, h_prev_l1
    __shared__ float h_prev_l0[64];
    __shared__ float h1_smem[64];
    __shared__ float h_prev_l1[64];

    // 初始化 h_prev，如果 hx 提供则使用，否则零
    if (hx != nullptr) {
        // hx 形状 [num_layers, batch, hidden_size]
        h_prev_l0[hidden_unit] = hx[0 * batch_size * hidden_size + batch_idx * hidden_size + hidden_unit];
        h_prev_l1[hidden_unit] = hx[1 * batch_size * hidden_size + batch_idx * hidden_size + hidden_unit];
    } else {
        h_prev_l0[hidden_unit] = 0.0f;
        h_prev_l1[hidden_unit] = 0.0f;
    }
    __syncthreads();

    for (int t = 0; t < seq_length; ++t) {
        // 第一层计算
        float pre_act_l0 = bias_ih_l0[hidden_unit] + bias_hh_l0[hidden_unit]; // 两个 bias 相加
        const float* x_t = input + (t * batch_size + batch_idx) * input_size;
        for (int j = 0; j < input_size; ++j) {
            pre_act_l0 += x_t[j] * weight_ih_l0[hidden_unit * input_size + j];
        }
        for (int k = 0; k < hidden_size; ++k) {
            pre_act_l0 += h_prev_l0[k] * weight_hh_l0[hidden_unit * hidden_size + k];
        }
        float h_l0 = tanhf(pre_act_l0);
        h1_smem[hidden_unit] = h_l0;
        __syncthreads();

        // 第二层计算
        float pre_act_l1 = bias_ih_l1[hidden_unit] + bias_hh_l1[hidden_unit];
        for (int k = 0; k < hidden_size; ++k) {
            pre_act_l1 += h1_smem[k] * weight_ih_l1[hidden_unit * hidden_size + k];
        }
        for (int k = 0; k < hidden_size; ++k) {
            pre_act_l1 += h_prev_l1[k] * weight_hh_l1[hidden_unit * hidden_size + k];
        }
        float h_l1 = tanhf(pre_act_l1);
        output[(t * batch_size + batch_idx) * hidden_size + hidden_unit] = h_l1;
        // 更新 h_prev
        h_prev_l0[hidden_unit] = h_l0;
        h_prev_l1[hidden_unit] = h_l1;
        __syncthreads(); // 确保所有线程完成更新，下一时间步读取一致
    }

    // 保存最终隐藏状态
    h_n[0 * batch_size * hidden_size + batch_idx * hidden_size + hidden_unit] = h_prev_l0[hidden_unit];
    h_n[1 * batch_size * hidden_size + batch_idx * hidden_size + hidden_unit] = h_prev_l1[hidden_unit];
}

// Launcher 函数，供 binding.cpp 调用
extern "C" void rnn_forward_launcher(
    const float* input,
    const float* weight_ih_l0,
    const float* weight_hh_l0,
    const float* bias_ih_l0,
    const float* bias_hh_l0,
    const float* weight_ih_l1,
    const float* weight_hh_l1,
    const float* bias_ih_l1,
    const float* bias_hh_l1,
    const float* hx,
    float* output,
    float* h_n,
    int seq_length,
    int batch_size,
    int input_size,
    int hidden_size,
    int num_layers,
    cudaStream_t stream
) {
    // 检查尺寸是否符合固定 kernel
    if (input_size != 100 || hidden_size != 64 || num_layers != 2) {
        // 可以返回错误或调用通用 kernel，但这里简单返回，由 binding 层检查。
        // 实际 binding 会检查这些，所以这里不应发生。
        return;
    }

    // 启动 kernel：每个 batch 一个 block，block 内 64 线程
    dim3 grid(batch_size);
    dim3 block(64);
    rnn_forward_fixed<<<grid, block, 0, stream>>>(
        input, weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0,
        weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1,
        hx, output, h_n, seq_length, batch_size
    );
}
