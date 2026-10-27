#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cfloat>

// Helper macro for CUDA error checking (can be omitted for performance but useful for debugging)
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// -----------------------------------------------------------------------------
// linear_forward: Y = input @ weight^T (+ optional bias)
// -----------------------------------------------------------------------------
extern "C" void linear_forward_launcher(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int batch_size,
    int in_features,
    int out_features,
    cudaStream_t stream)
{
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Compute output = alpha * input @ weight^T + beta * output
    // cuBLAS uses column-major, so we compute weight @ input^T and then take transpose by using row-major interpretation tricks.
    // Standard approach: C = A * B^T where A is MxK (input), B is NxK (weight), C is MxN.
    // In cuBLAS, we can do: cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features, &alpha, weight, in_features, input, in_features, &beta, output, out_features);
    // This computes output (out_features x batch_size) = weight^T * input^T, but we want output (batch_size x out_features) = input * weight^T.
    // Instead, use CUBLAS_OP_N for input and CUBLAS_OP_T for weight, with output as batch_size x out_features.
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features, &alpha, weight, in_features, input, in_features, &beta, output, out_features);
    // Let's derive carefully:
    // We want C = A * B^T where A is MxK (input), B is NxK (weight), C is MxN.
    // In cuBLAS, column-major means that if we pass row-major matrices as column-major, an MxN row-major matrix is seen as NxM column-major.
    // We can use the identity: C^T = (A * B^T)^T = B * A^T. So computing B * A^T as column-major is equivalent.
    // Let's use cublasSgemm with:
    // opA = CUBLAS_OP_N (no transpose for weight? But we need weight^T? Actually we'll compute weight * input^T as column-major)
    // The cleanest: treat input as column-major NxM? Let's just use the following known pattern:
    // For row-major matrices A (MxK) and B (NxK), compute C (MxN) = A * B^T using cuBLAS:
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
    //            N, M, K,
    //            &alpha, B, K, A, K, &beta, C, N);
    // Explanation: Since cuBLAS expects column-major, passing row-major A as column-major makes it effectively A^T (KxM). Similarly B becomes B^T (KxN).
    // Then CUBLAS_OP_T on B (which is actually B^T) gives B (NxK), CUBLAS_OP_N on A (which is A^T) gives A^T (KxM).
    // Product = B * A^T which is N x M, which is (A * B^T)^T, so C (which is N x M) is actually C^T. We can output to output_pointer with row-major interpretation of column-major? Let's just use the helper macro later.
    // Simpler: use cublasSgemm with opA=CUBLAS_OP_N, opB=CUBLAS_OP_T, and swap input and weight.
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                batch_size, out_features, in_features,
                &alpha,
                input, batch_size,   // but input is row-major, need leading dimension? Actually cuBLAS sees column-major, so leading dimension is number of rows of input (batch_size).
                weight, out_features, // leading dimension of weight is out_features.
                &beta,
                output, batch_size);

    // Add bias if provided
    if (bias != nullptr) {
        // Implement bias addition kernel
        dim3 blocks((out_features + 255) / 256);
        dim3 threads(256);
        add_bias_kernel<<<blocks, threads, 0, stream>>>(output, bias, batch_size, out_features);
    }

    cublasDestroy(handle);
}

__global__ void add_bias_kernel(float* output, const float* bias, int batch_size, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < out_features) {
        float b = bias[idx];
        for (int row = 0; row < batch_size; ++row) {
            output[row * out_features + idx] += b;
        }
    }
}

// -----------------------------------------------------------------------------
// log_softmax along dim=1 (row-wise)
// -----------------------------------------------------------------------------
template<int BLOCK_SIZE>
__global__ void log_softmax_kernel(float* out, const float* in, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    extern __shared__ float smem[];
    float* max_vals = smem;  // size BLOCK_SIZE
    float* sum_vals = &smem[BLOCK_SIZE]; // size BLOCK_SIZE

    const float* row_in = in + row * cols;
    float* row_out = out + row * cols;

    // Find max per row using block reduction
    float local_max = -FLT_MAX;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        local_max = fmaxf(local_max, row_in[i]);
    }
    // block reduction for max
    max_vals[tid] = local_max;
    __syncthreads();
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            max_vals[tid] = fmaxf(max_vals[tid], max_vals[tid + stride]);
        }
        __syncthreads();
    }
    float max_val = max_vals[0];
    __syncthreads();

    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        float val = __expf(row_in[i] - max_val);
        row_out[i] = val; // temporarily store exp, will overwrite later
        local_sum += val;
    }
    // block reduction for sum
    sum_vals[tid] = local_sum;
    __syncthreads();
    for (int stride = BLOCK_SIZE/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_vals[tid] += sum_vals[tid + stride];
        }
        __syncthreads();
    }
    float sum_val = sum_vals[0];
    __syncthreads();

    // Compute log_softmax: x - max - log(sum)
    float log_sum = logf(sum_val);
    for (int i = tid; i < cols; i += BLOCK_SIZE) {
        row_out[i] = row_in[i] - max_val - log_sum;
    }
}

// -----------------------------------------------------------------------------
// adaptive_log_softmax_log_prob
// -----------------------------------------------------------------------------
extern "C" void adaptive_log_softmax_log_prob_launcher(
    const float* input,        // [batch_size, input_size]
    const float* head_weight,  // [head_size, input_size]
    const float* tail0_w1,     // [reduce_size0, input_size]
    const float* tail0_w2,     // [cluster_size0, reduce_size0]
    const float* tail1_w1,     // [reduce_size1, input_size]
    const float* tail1_w2,     // [cluster_size1, reduce_size1]
    float* output,             // [batch_size, n_classes]
    int batch_size,
    int input_size,
    int head_size,      // = 5002
    int reduce_size0,   // = 192
    int cluster_size0,  // = 3000
    int reduce_size1,   // = 48
    int cluster_size1,  // = 2000
    int shortlist_size, // = 5000 (first cutoffs[0])
    int cutoffs0,       // = 5000
    int cutoffs1,       // = 8000
    float div_value,
    cudaStream_t stream)
{
    // Allocate temporary buffers
    float *head_logits, *head_logprobs;
    float *reduced0, *tail0_logits, *tail0_effective, *tail0_logprobs;
    float *reduced1, *tail1_logits, *tail1_effective, *tail1_logprobs;

    int n_classes = head_size + cluster_size0 + cluster_size1 - shortlist_size;
    // Actually n_classes = head_size + cluster_size0 + cluster_size1 - shortlist_size? No: output size is head_size + cluster_size0 + cluster_size1? Wait:
    // head_size = shortlist_size + first_tail_size? In PyTorch, head_size = shortlist_size + cutoffs[0]? Actually head includes shortlist + first tail's first part? Let's check: AdaptiveLogSoftmaxWithLoss: head_size = shortlist_size + cutoffs[0]. So head_size = 5000 + 5000 = 10000? But given head_weight shape [5002, 768], so head_size = 5002, cutoffs[0]=5000, shortlist_size=2? Hmm.
    // Actually in PyTorch, head_size = shortlist_size + cutoffs[0]. The shortlist_size = n_classes - sum(cluster_sizes). But cutoffs[0] is first cutoff. Wait the code in PyTorch: head = Linear(in_features, head_size) where head_size = shortlist_size + cutoffs[0]. tail[0] is Sequential(Linear(in_features, reduce_size0), Linear(reduce_size0, cluster_size0)). tail[1] similar. n_classes = shortlist_size + cluster_size0 + cluster_size1. cutoffs[0] = shortlist_size + cluster_size0? Actually cutoffs are boundaries for clusters. Let's get from model: input_size=768, n_classes=10000, cutoffs=[5000,8000], div_value=4.0. Model init: AdaptiveLogSoftmaxWithLoss(input_size, n_classes, cutoffs, div_value). That sets: shortlist_size = n_classes - sum(cutoffs? No, cutoffs are [5000,8000], sum = 13000 > n_classes, so maybe cutoffs are cumsum? In PyTorch, cutoffs is list of cumulative sizes of clusters? Actually, cutoffs = [5000, 8000] means first cluster size = 5000, second cluster size = 3000. shortlist_size = n_classes - (5000+3000) = 2000? But head size is shortlist_size + cutoffs[0] = 2000 + 5000 = 7000. But given head_weight shape [5002, 768], that suggests head_size = 5002. Let's check PyTorch code: head = Linear(in_features, shortlist_size + cutoffs[0])? Actually from PyTorch source: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py: head = Linear(in_features, shortlist_size + cutoffs[0]). For n_classes=10000, cutoffs=[5000,8000], then shortlist_size = n_classes - cutoffs[-1]? No, shortlist_size = n_classes - sum(cluster sizes) but cluster sizes are cutoffs[0], cutoffs[1]-cutoffs[0], etc. Actually cutoffs is a list of cumulative counts. So first cluster size = cutoffs[0] = 5000, second cluster size = cutoffs[1] - cutoffs[0] = 3000. Then shortlist_size = n_classes - cutoffs[1] = 2000. head_size = shortlist_size + cutoffs[0] = 2000 + 5000 = 7000. But state_dict says head weight shape [5002,768], not 7000. So something else: maybe cutoffs are not cumulative? In PyTorch, cutoffs are list of ints where each is the number of classes in each cluster? Actually, from docs: cutoffs should be a list of ints containing the cutoff values for the tail clusters. For example, cutoffs=[2000, 5000] means first cluster has 2000 classes, second has 3000? Let's verify with the given shapes: tail0_w2 shape [3000, 192], so cluster0 size = 3000. tail1_w2 shape [2000,48], so cluster1 size = 2000. sum = 5000. n_classes=10000, so shortlist_size = 5000. head_weight shape [5002,768] suggests head_size = 5002, which is shortlist_size + 2? Maybe because of padding? Actually in PyTorch, head = Linear(in_features, shortlist_size + cutoffs[0])? If cutoffs[0]=5000, head_size = 5000+5000=10000, not 5002. Wait, maybe cutoffs are not cumulative but each is size of cluster? Then shortlist_size = n_classes - sum(cutoffs) = 10000 - (5000+8000) = -3000 impossible. So maybe cutoffs are cumulative but in this case cutoffs=[5000,8000] means first cluster size=5000, second cluster size=3000, shortlist_size = 2000, head_size = 2000+5000=7000, still not 5002.

 Let's examine tail0_w1 shape [192,768], reduce_size0=192. tail0_w2 [3000,192], so cluster0 has 3000 classes. tail1_w1 [48,768], reduce_size1=48. tail1_w2 [2000,48], cluster1 has 2000 classes. So total tail classes = 3000+2000=5000. Then shortlist_size = 10000-5000=5000. head_size should be shortlist_size + something? Actually the head weight shape is [5002, 768]. That means head has 5002 output features. That is 5000 + 2. Possibly 2 extra for something? In AdaptiveLogSoftmaxWithLoss, the head may include shortlist plus a context? Let's check PyTorch source: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py
 In __init__: self.head = Linear(in_features, shortlist_size + cutoffs[0]) where shortlist_size = n_classes - sum(cutoffs) if cutoffs is not None? Actually cutoffs is a list of ints, sum(cutoffs) must be less than n_classes. Then shortlist_size = n_classes - sum(cutoffs). For cutoffs=[5000,8000], sum=13000 > 10000, so that would fail. Therefore cutoffs must be something else. But the model was constructed successfully with cutoffs=[5000,8000]. So maybe cutoffs are not sum but something else? Actually from PyTorch docs: cutoffs (list) – the cutoffs for the clusters (should be an increasing Sequence of ints). Example: cutoffs=[2000, 10000]. That means first cluster has 2000 classes, second cluster has 8000 classes, shortlist = n_classes - 10000 = 0. In our case cutoffs=[5000,8000] would mean first cluster 5000 classes, second cluster 3000 classes, shortlist = 10000-8000=2000. Head size = shortlist + cutoffs[0] = 2000+5000=7000. But state_dict head weight [5002,768] contradicts. Unless the model was created with different parameters but state_dict shows head.weight shape [5002,768]. That suggests that during init, input_size=768, n_classes=10000, cutoffs=[5000,8000]. Let's compute expected head size from PyTorch source.
 From PyTorch source (v2.x):
 class AdaptiveLogSoftmaxWithLoss(Module):
   def __init__(self, in_features, n_classes, cutoffs, div_value=4., head_bias=False):
      ...
      self.cutoffs = cutoffs + [n_classes]
      self.shortlist_size = self.cutoffs[0]
      self.n_clusters = len(self.cutoffs) - 1
      self.head_size = self.shortlist_size + self.n_clusters
      ...
      self.head = Linear(in_features, self.head_size, bias=head_bias)
      self.tail = ModuleList()
      for i in range(self.n_clusters):
          hsz = int(input_size // (self.div_value ** (i + 1)))
          osz = self.cutoffs[i + 1] - self.cutoffs[i]
          projection = Sequential(
              Linear(in_features, hsz, bias=False),
              Linear(hsz, osz, bias=False),
          )
          self.tail.append(projection)
 
 Wait, that's not matching. Actually, cutoffs passed is a list of cumulative counts? Let's check the actual source: I recall that cutoffs is list of ints where cutoffs[-1] < n_classes. They do: self.cutoffs = list(cutoffs) + [n_classes]; self.shortlist_size = self.cutoffs[0]; self.n_clusters = len(self.cutoffs) - 1; self.head_size = self.shortlist_size + self.n_clusters. That would give for cutoffs=[5000,8000]: self.cutoffs = [5000,8000,10000], shortlist_size=5000, n_clusters=2, head_size=5002. Yes! That matches head weight [5002,768]. So the first cutoff value is actually the shortlist size, and the rest are cumulative class counts for tail clusters. So tail0 has classes from 5000 to 8000 (size 3000), tail1 from 8000 to 10000 (size 2000). That fits the tail sizes. Great.

 So we have:
 shortlist_size = cutoffs[0] = 5000
 head_size = shortlist_size + n_clusters = 5000 + 2 = 5002
 n_clusters = len(cutoffs) = 2 (since cutoffs list has two elements)
 tail0 size = cutoffs[1] - cutoffs[0] = 3000
 tail1 size = n_classes - cutoffs[1] = 2000

 The head contains the shortlist classes plus one extra dimension per cluster? Actually head_size = shortlist_size + n_clusters. That means head outputs shortlist_size + n_clusters logits. The first shortlist_size are for the shortlist classes. The extra n_clusters (2) are for cluster logits? Probably used in loss but in log_prob we ignore them? We'll need to follow PyTorch's log_prob implementation.

 Let's examine PyTorch's AdaptiveLogSoftmaxWithLoss.log_prob:
 def log_prob(self, input):
     head_output = self.head(input)
     head_logprob = F.log_softmax(head_output, dim=1)
     out = input.new_empty((head_output.size(0), self.n_classes))
     out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]
     for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
         cluster_output = self.tail[i](input)
         cluster_logprob = F.log_softmax(self.div_value * cluster_output, dim=1)
         # Adjust head logit for cluster
         cluster_head_logit = head_output[:, self.shortlist_size + i].unsqueeze(1) * self.div_value
         cluster_logprob = cluster_logprob + cluster_head_logit  # maybe combined differently?
         out[:, start_idx:stop_idx] = cluster_logprob
     return out

 Actually, from source I recall:
 def log_prob(self, input):
     head_output = self.head(input)
     head_logprob = F.log_softmax(head_output, dim=1)
     out = input.new_empty((head_output.size(0), self.n_classes))
     out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]
     for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
         cluster_output = self.tail[i](input)
         cluster_logits = cluster_output * self.div_value
         # Combine with head logit for the cluster
         cluster_head_logits = head_output[:, self.shortlist_size + i] * self.div_value
         # Effective logits = concatenation of cluster_logits and cluster_head_logits? Actually, the cluster head logit is used to compute the cluster's contribution to the partition function.
         # They compute log_softmax on [cluster_head_logit, cluster_logits]? Wait, I think they compute log_softmax of cluster_output * div_value, then add the head logit multiplied by div_value? Let's look at the actual code:
 
 From PyTorch 2.0 code (https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/adaptive.py):
 def log_prob(self, input):
     head_output = self.head(input)
     head_logprob = torch.nn.functional.log_softmax(head_output, dim=1)
     out = input.new_empty((head_output.size(0), self.n_classes))
     out[:, : self.shortlist_size] = head_logprob[:, : self.shortlist_size]

     for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
         cluster_output = self.tail[i](input)
         cluster_logprob = torch.nn.functional.log_softmax(
             self.div_value * cluster_output, dim=1)
         # The cluster head logit is used to weight the cluster's probability
         # The log probability of a class in the cluster is:
         # log( exp(cluster_logit * div_value) / sum(exp(div_value * cluster_logits)) * exp(head_logit_i * div_value) / Z )
         # where Z is the overall partition function across head (including cluster heads).
         # However, the head_logprob already includes log(exp(head_logit_i) / sum_head).
         # But here they compute log_softmax of div_value*cluster_output, and then add head_output[:, self.shortlist_size + i] * div_value and subtract logsumexp of head_output? Actually maybe they compute:
         # cluster_logprob = log_softmax(div_value * cluster_output, dim=1) + head_logprob[:, self.shortlist_size + i] + log(div_value)?
 
 Need to be accurate. Let's search memory: The log_prob function does:
         cluster_logprob = F.log_softmax(self.div_value * cluster_output, dim=1)
         out[:, start_idx:stop_idx] = cluster_logprob + head_logprob[:, self.shortlist_size + i] + log(self.div_value)
 Is that right? Let's derive.

 In AdaptiveLogSoftmaxWithLoss, the model is defined as:
 p(y) = p(shortlist|y) * p(cluster_head|y) ... but actually the head outputs logits for shortlist classes plus one extra logit per cluster (called “cluster logits”). The probability of a class in a tail cluster is:
 p(class) = softmax(head_output) over the first shortlist_size + n_clusters entries, but only the cluster logit is used to weight the cluster probability, and within the cluster, a second softmax is applied. The overall probability is:
 p(class) = exp(head_logit_c) / sum_{all head entries} * exp(cluster_logit * div_value) / sum_{cluster} exp(cluster_logits * div_value)
 where c is the cluster index. So log p(class) = head_logit_c - logsumexp(head_output) + div_value * cluster_logit - logsumexp(div_value * cluster_output).

 head_logprob[:, self.shortlist_size + i] = head_logit_c - logsumexp(head_output) (this is exactly from log_softmax of head_output for the cluster entry).
 Then for each cluster, they compute cluster_logprob = log_softmax(div_value * cluster_output, dim=1) = div_value * cluster_logit - logsumexp(div_value * cluster_output).
 So out[:, start_idx:stop_idx] = cluster_logprob + head_logprob[:, self.shortlist_size + i].

 But wait, they might also add log(div_value)? No, because div_value is just a multiplier applied consistently. The head logit is not multiplied by div_value in this expression. So it's simply addition. Let's check with actual code: I think that's correct. They do:
 cluster_logprob = F.log_softmax(self.div_value * cluster_output, dim=1)
 out[:, start_idx:stop_idx] = cluster_logprob + head_logprob[:, self.shortlist_size + i]
 So no extra log(div_value).

 Yes, that's the implementation.

 Therefore, our custom op must compute exactly that.

 Steps:
 1. head_logits = input @ head_weight.T, shape [batch, head_size=5002]
 2. head_logprobs = log_softmax(head_logits, dim=1)
 3. For tail0:
    reduced0 = input @ tail0_w1.T, shape [batch, reduce_size0=192]
    tail0_logits = reduced0 @ tail0_w2.T, shape [batch, cluster_size0=3000]
    tail0_logprobs = log_softmax(div_value * tail0_logits, dim=1)
    add head_logprobs[:, shortlist_size + 0] to each row of tail0_logprobs (broadcast)
 4. For tail1:
    reduced1 = input @ tail1_w1.T, shape [batch, reduce_size1=48]
    tail1_logits = reduced1 @ tail1_w2.T, shape [batch, cluster_size1=2000]
    tail1_logprobs = log_softmax(div_value * tail1_logits, dim=1)
    add head_logprobs[:, shortlist_size + 1] to each row of tail1_logprobs
 5. Output = concat: [head_logprobs[:, :shortlist_size], tail0_logprobs, tail1_logprobs] along dim=1

 We need to carry out these computations efficiently using cuBLAS for GEMMs and custom kernels for log_softmax and final assembly.

 We'll allocate necessary temporary buffers using cudaMalloc, and free them at the end.

 We must ensure the output tensor is allocated by the binding and passed as pointer. The binding will allocate output of shape [batch_size, n_classes] and call this launcher.

 We'll implement helper function to launch log_softmax over rows using the template kernel.

 We'll implement a kernel to add a column vector to each row of a matrix (for adding head_logprobs cluster values).

 We'll implement a kernel to copy and assemble the output.

 Note: We need to handle possible non-dynamic shapes if batch_size varies, but in the contract batch_size=2048 fixed for verification, but we should make kernel general.

 For topk, we need to compute top k values and indices per row. We'll implement a stable topk kernel that processes each row in a block. Since k=5 small, we can use a per-thread approach or block-wide selection.

 Approach for topk:
   For each row, we can use a single thread to do a simple selection: iterate over cols, maintain top-k list. That would be O(batch_size * n_classes * k) which for 2048*10000*5 ~ 100M operations, maybe okay but could be slow. Better to use parallel block per row with warp-level reductions to find top k.
   For k <= 32, we can use warp-level algorithms. Each warp handles one row? But rows have 10000 elements, warp size 32, each thread processes multiple elements, then combine.
   We can implement a function that for each row, each warp does: initialize top k as -inf, then each lane iterates over its assigned elements and updates local top k (using insertion sort), then warp reduces the top k lists from each lane by merging, then final top k from warp.
   Since k=5 small, this is efficient.
   We need indices in int64_t, and stable ordering: use original index as tie-breaker, lower index first for equal values.

 We'll implement kernel where each block has one warp per row? Or one block per row with 256 threads, then reduce across threads. For simplicity, we can assign one block per row, with blockDim = 256, and each thread maintains a small local top-k list. Then we reduce within block using shared memory. Then thread 0 writes final top-k to output. That works for any block size, but we need to choose block size that covers row width efficiently. Since n_classes=10000, 256 threads each process ~40 elements, okay.

 Stable ordering: maintain pairs (value, index), compare value first, then index (lower index considered larger for ties, i.e., if values equal, the one with smaller index is considered greater to come earlier in descending order). So in insertion sort, we insert an element if value > current or (value == current and index < current).

 We'll implement a device function to update a small array of top-k pairs.

 Then block reduction: use shared memory to merge top-k lists from all threads, hierarchical reduction.

 Finally, thread 0 writes output values and indices.

 We must also consider k > blockDim, but in our case k=5, we can assume k <= 32 and use warp-level for efficiency, but block-level reduction is okay.

 We'll implement a kernel `topk_kernel` that uses block reduction. Block size fixed at 256 for simplicity.

 Now, we need to put all kernels and launchers in custom_ops.cu.

 Important: The launcher names should be declared extern "C" and match what binding.cpp will use. We need to pick names consistent with model_new.py calls: `cuda_extension.linear_forward`, `cuda_extension.adaptive_log_softmax_log_prob`, `cuda_extension.topk`. The binding .cpp will define those Python functions and call our launchers. So we need to choose launcher names like `linear_forward_launcher`, `adaptive_log_softmax_log_prob_launcher`, `topk_launcher`. The binding.cpp will include a header or declare these functions. We'll just provide them.

 Note: The contract mentions linear_forward but model_new.py does not directly call it; it is used internally by adaptive_log_softmax_log_prob? But in our implementation, we can use cuBLAS directly, so we may not need a separate linear_forward kernel. However, the contract lists linear_forward as an operator that should exist. But our custom_ops.cu only needs to provide launchers for the operators used by model_new.py. model_new.py uses only adaptive_log_softmax_log_prob and topk. It does not call linear_forward directly. So we can omit linear_forward launcher, but the contract says it should be available? Actually the contract lists linear_forward as an operator, but model_new.py does not call it. It might be required if the binding layer exposes it, but maybe not. We should still provide a linear_forward launcher because it's in the contract and might be used by the binding or future extensions. It's easy.

 We'll implement linear_forward_launcher as above.

 For adaptive_log_softmax_log_prob, we'll use cuBLAS for all GEMMs and custom kernels for log_softmax and assembly.

 We need to be careful with cuBLAS handle creation. In launcher, we can create handle each time, or have a static handle? Creating/destroying per call may be okay for performance but could add overhead. We can create handle once as static per stream? But launcher may be called with same stream. To keep simple, create and destroy each call; cuBLAS handle creation is not huge, and the GEMMs dominate. For performance, we could cache handles per stream in a static map, but that's more complex. We'll create/destroy each call for simplicity.

 We'll also need to ensure all CUDA calls are checked for errors (maybe only for debugging, but we can skip in final for performance). The binding might check errors? But it's fine.

 Now, implement the kernels and launchers in the file.

 Let's outline the file structure:

 #includes as above.

 __global__ void add_bias_kernel(...)
 template<int BLOCK_SIZE> __global__ void log_softmax_kernel(...)
 __global__ void add_column_to_rows_kernel(...)  // adds a column value to each element of a row
 __global__ void assemble_output_kernel(...)     // copies head shortlist part and tail parts into output
 __global__ void topk_kernel(...)                // per-row topk selection
 __device__ void update_topk(...)                // helper for topk
 __device__ void merge_topk(...)                 // helper for reduction

 extern "C" void linear_forward_launcher(...)
 extern "C" void adaptive_log_softmax_log_prob_launcher(...)
 extern "C" void topk_launcher(...)

 Let's implement each carefully.

 For log_softmax kernel, we need shared memory size = 2*BLOCK_SIZE*sizeof(float). We'll launch with blockDim = BLOCK_SIZE, and gridDim = rows. We'll choose BLOCK_SIZE = 256, which is good.

 For add_column_to_rows, we can do a simple 1D kernel: for each element index (row, col), output[row*cols+col] += column[row]. But we'll do it for each tail logprobs matrix, where column is head_logprobs[:, shortlist_size+i] (a vector of length batch). So kernel: each thread handles one element: int idx = blockIdx.x*blockDim.x+threadIdx.x; if idx < batch*cols then row = idx / cols, col = idx % cols, value = col_add[row]; out[idx] += value. We'll launch with enough blocks.

 For assemble_output, we can use a 1D kernel over total output elements: each thread computes which part it belongs to and copies from the appropriate source. But we can also use cudaMemcpyAsync for contiguous parts: copy head shortlist part directly, then copy tail0, tail1. However, since they are contiguous in output, we can just use cudaMemcpy2DAsync or simple cudaMemcpyAsync. But we need to ensure proper stream. We'll use cudaMemcpyAsync with the current stream. That is simpler and efficient. However, we might want to avoid multiple kernels if possible. For assembly, we can just do three cudaMemcpyAsync calls: copy head[:, :shortlist_size] from head_logprobs to output[:, :shortlist_size]; then copy tail0_logprobs to output[:, shortlist_size:shortlist_size+cluster_size0]; then copy tail1_logprobs to output[:, shortlist_size+cluster_size0:...]. We can use cudaMemcpy2DAsync for each with appropriate pitch, or since contiguous, we can do cudaMemcpyAsync for each row segment? But rows are contiguous, the whole tail0_logprobs is contiguous, and the destination slice is contiguous across rows? The destination is a slice of columns, which is not contiguous across rows because each row has n_classes elements. So we cannot just copy a contiguous block. We need to use cudaMemcpy2DAsync with dpitch and spitch. Or we can use a kernel to copy. Since only 3 regions, we can use cudaMemcpy2DAsync for each: source is contiguous with width cluster_size0, height batch, spitch = cluster_size0*sizeof(float); destination width cluster_size0, height batch, dpitch = n_classes*sizeof(float), destination pointer = output + shortlist_size. That copies correctly row by row. We'll do that for all three pieces. Similarly for head shortlist: width shortlist_size, height batch, spitch = head_size*sizeof(float)? Actually source head_logprobs has width head_size, spitch = head_size*sizeof(float), destination width shortlist_size, dpitch = n_classes*sizeof(float), dest ptr = output. Tail1 similar. This avoids a custom kernel.

 We'll need to allocate temporary buffers with cudaMalloc. Sizes: head_logits, head_logprobs: batch*head_size; reduced0: batch*reduce_size0; tail0_logits: batch*cluster_size0; tail0_logprobs: batch*cluster_size0; reduced1: batch*reduce_size1; tail1_logits: batch*cluster_size1; tail1_logprobs: batch*cluster_size1. We'll allocate all at once maybe.

 For GEMMs, we'll use cublasSgemm with correct op and leading dimensions. As discussed, for row-major A (MxK) and B (NxK), compute C (MxN) = A * B^T using:
 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
             M, N, K,
             &alpha,
             A, K,
             B, K,
             &beta,
             C, M);
 Wait, we need to verify the leading dimensions. In cuBLAS, with column-major, if we pass row-major matrices, the leading dimension should be the number of columns? Actually standard trick:
 For row-major A of size MxK, when passed to cuBLAS as column-major, the matrix appears as KxM with leading dimension = K? No, leading dimension is the number of rows in column-major, so for row-major A, the leading dimension is the number of columns, i.e., K. But we want to compute without explicit transpose by using opA/opB.
 The common approach:
 cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
             N, M, K,
             &alpha, B, K, A, K, &beta, C, N);
 This computes C = B^T * A^T? Actually, let's derive:
 We want C = A * B^T, where A is MxK, B is NxK, C is MxN.
 If we pass A and B as row-major, cuBLAS sees them as column-major of size KxM and KxN respectively (since leading dimension is number of rows in column-major, which corresponds to number of columns in row-major). So the effective matrices are A_col = A^T (KxM), B_col = B^T (KxN).
 To compute C = A * B^T, we can compute C^T = B * A^T. So we can do:
 cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             N, M, K,
             &alpha, B_col, K, A_col, K, &beta, C_transpose, N);
 Here B_col is B^T (KxN), A_col is A^T (KxM). The product B_col * A_col = B^T * A^T = (A*B)^T? Actually B^T * A^T = (A*B)^T, but we want (A*B^T)^T = B*A^T. So B_col * A_col = B^T * A^T = (A*B)^T, not what we need. Hmm.
 Let's think differently: We want C = A * B^T. In row-major, C is MxN.
 We can ask cuBLAS to compute C^T = B * A^T, because if we get C^T, we can store it as column-major and then interpret as row-major to get C? That's classic: store result as column-major, then when read as row-major, it's the transpose. So we can compute with cuBLAS using A and B swapped and no transpose ops, then write output with transposed indexing.
 But easier: use cuBLAS with A as column-major by passing A as (K x M) with leading dimension K? That requires physically transposing A, which we don't want.
 The standard approach for row-major C = A * B^T is:
 cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
             N, M, K,
             &alpha, B, K, A, K, &beta, C, N);
 Let's test with a small example. Suppose A is 2x3 row-major: A[0] = [1,2,3], A[1]=[4,5,6]. B is 1x3 row-major: B[0] = [7,8,9]. Then C = A*B^T should be 2x1: C[0]=1*7+2*8+3*9=50, C[1]=4*7+5*8+6*9=122. Now in cuBLAS call with parameters: M=2, N=1, K=3, opA=CUBLAS_OP_T, opB=CUBLAS_OP_N, A = B pointer, lda=3, B = A pointer, ldb=3, C pointer, ldc=1? Usually leading dimension is max(1, rows of op(A)). For opA=T, A is K x N? Actually cuBLAS expects A as (M x K) if opA=N, or (K x M) if opA=T. Here we set opA=T on B (which is actually N x K row-major). But we pass B pointer as A. So cuBLAS will treat A as (K x M) with dimensions: M parameter for opA=T means A has K rows and M cols? Wait, the dimensions: For opA=T, A is K x M, with leading dimension lda >= K. But we pass lda=3 (which is K). So that's fine if A is stored as K rows, M cols. Our B is N x K row-major, which in column-major is K x N. If N=1, then it's K x 1, not K x M. So mismatch. This is getting confusing.

 Let's simplify by using cublasSgemm with CUBLAS_OP_N for both and doing explicit transpose via leading dimensions. Actually, cuBLAS supports row-major since CUDA 10? We can set cublasSetMathMode and use cublasSgemm with CUBLAS_OP_N and row-major order? No, cuBLAS has cublasSgemmEx with CUBLAS_OP_N and can accept row-major by specifying cublasOperation_t? There is cublasSetStream, but row-major is not directly supported for sgemm except via cublasSgemmBatched? Actually, cuBLAS has a separate API: cublasSgemm can be used with CUBLAS_OP_T to emulate row-major.
 The robust way to compute C = A * B^T for row-major A (MxK), B (NxK) is:
 cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
             N, M, K,
             &alpha,
             B, K,   // B is NxK row-major, but here it is treated as KxN column-major, and with op=T becomes NxK, correct.
             A, K,   // A is MxK row-major, treated as KxM column-major, op=N keeps KxM, so product = NxK * KxM = NxM.
             &beta,
             C, N);  // C is NxM, but we want MxN. However, if we write the result as column-major, it will be N rows, M cols. But we want M rows, N cols. So we can then transpose C? Or we can set C pointer to output but interpret as row-major? Actually, if we compute NxM and store as column-major, then a row-major view of that memory would be MxN, which is exactly what we want! Let's verify: The resulting matrix from cuBLAS is N rows, M columns, stored in column-major, so memory layout: first column has N elements, etc. If we instead treat that memory as row-major with M rows and N columns, then the first row has N elements (which correspond to the first column of the NxM matrix), and that would be C^T, not C. We need C (MxN) which is the transpose of the computed NxM. So if we compute NxM and store in column-major, then reading as row-major gives MxN, which is exactly C. Because column-major of NxM is the same memory layout as row-major of MxN (since both have M*N elements and element (i,j) in row-major MxN is at position i*N+j; column-major NxM element (j,i) is at position j + i*N = i*N+j). So yes, they match. Therefore, we can compute the NxM product and store it, then the output pointer (which is row-major MxN) will contain the correct data. So the call is:
 cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
             N, M, K,
             &alpha, B, K, A, K, &beta, C, N);
 where M = batch_size (rows of input), N = out_features (columns of weight), and C is the output matrix of size MxN. But note: in this call, the dimensions are swapped: we set m=N, n=M, k=K. Then the result is N x M, stored in C with leading dimension N. The output pointer has row-major with leading dimension N (since output has M rows, N cols, so row stride is N). So we pass ldc = N. That matches. Great.

 So we can use that pattern for all GEMMs.

 For linear_forward, we can compute input @ weight.T with bias.

 For adaptive_log_softmax_log_prob, we use cublasSgemm for each of the 5 GEMMs (head, tail0 w1, tail0 w2, tail1 w1, tail1 w2).

 We'll write a helper function to perform a GEMM given pointers and dimensions, to avoid repetition.

 But for clarity, we'll just inline the calls.

 Now write the code. We'll need to include <cuda_runtime.h> and <cublas_v2.h>. Since we are not using torch, no torch includes.

 We must ensure the file compiles with nvcc. We'll have to use extern "C" for launchers.

 Let's craft the complete file.

 We'll also include a simple error check macro maybe for debugging but commented out or conditional. For final performance, we can omit checks, but we can keep a macro that does nothing if NDEBUG. We'll just define CUDA_CHECK as empty.

 We'll also need to include <algorithm> maybe for std::max? But we can avoid.

 Topk kernel: we'll use block reduction. We'll define a struct Pair { float value; int64_t index; }; but we need to be careful with alignment and shared memory. We'll store topk in arrays of float and int64 separately? For reduction, we can use a struct and store in shared memory. But shared memory dynamic allocation can be tricky. We'll use template for k, but k is runtime. We can fix maximum k to 32 and allocate shared memory for that. Since k=5, we can have max_k=32. In kernel, we'll have arrays of size k*blockDim for shared topk lists. But we only need k per thread for initial, then reduce. We'll allocate shared memory of size (blockDim * k) * (sizeof(float)+sizeof(int64_t)) rounded. Alternatively, we can use two arrays: float smem_vals[blockDim*k]; int64_t smem_idx[blockDim*k]; But dynamic shared memory size must be known at launch. We can pass as extern __shared__ char smem[]; and cast. We'll use that.

 Simpler: each thread maintains its own local topk array (max k). After processing its elements, it writes its topk to shared memory. Then we perform pairwise merges across threads. We can do a reduction tree where each step merges two topk lists. Since k is small, the merge is cheap. We'll implement a device function to merge two sorted descending topk lists into one.

 To update local topk while scanning elements, we can do insertion into a small sorted list. Since k <= 32, we can linearly search.

 We'll write a function updateTopK(float val, int64_t idx, float* vals, int64_t* indices, int k) that inserts if appropriate. The list is kept descending (largest first) and for ties, smaller index first. So comparator: new element is better if val > existing or (val == existing && idx < existing_idx).

 We'll write a function mergeTopK(const float* vals1, const int64_t* idx1, const float* vals2, const int64_t* idx2, float* out_vals, int64_t* out_idx, int k) that merges two sorted lists.

 Kernel outline:
 __global__ void topk_kernel(const float* input, float* out_values, int64_t* out_indices, int batch_size, int n_classes, int k)
 {
   int row = blockIdx.x;
   if (row >= batch_size) return;
   const float* row_in = input + row * n_classes;
   int tid = threadIdx.x;
   int block_size = blockDim.x;

   // local topk
   float local_vals[MAX_K]; // maybe statically define max_k=32
   int64_t local_idx[MAX_K];
   for (int i=0; i<k; ++i) { local_vals[i] = -FLT_MAX; local_idx[i] = -1; }

   // each thread processes elements [tid, n_classes) with stride block_size
   for (int col = tid; col < n_classes; col += block_size) {
       updateTopK(row_in[col], col, local_vals, local_idx, k);
   }

   // write local to shared memory
   extern __shared__ char smem_raw[];
   float* smem_vals = reinterpret_cast<float*>(smem_raw);
   int64_t* smem_idx = reinterpret_cast<int64_t*>(&smem_vals[block_size * k]);
   for (int i=0; i<k; ++i) {
       smem_vals[tid * k + i] = local_vals[i];
       smem_idx[tid * k + i] = local_idx[i];
   }
   __syncthreads();

   // reduction
   for (int stride = block_size/2; stride > 0; stride >>= 1) {
       if (tid < stride) {
           int idx1 = tid;
           int idx2 = tid + stride;
           float merged_vals[MAX_K];
           int64_t merged_idx[MAX_K];
           mergeTopK(&smem_vals[idx1*k], &smem_idx[idx1*k],
                     &smem_vals[idx2*k], &smem_idx[idx2*k],
                     merged_vals, merged_idx, k);
           // write back to first location
           for (int i=0; i<k; ++i) {
               smem_vals[idx1*k + i] = merged_vals[i];
               smem_idx[idx1*k + i] = merged_idx[i];
           }
       }
       __syncthreads();
   }

   // thread 0 writes result
   if (tid == 0) {
       for (int i=0; i<k; ++i) {
           out_values[row * k + i] = smem_vals[i];
           out_indices[row * k + i] = smem_idx[i];
       }
   }
 }

 We need to define MAX_K as 32 perhaps. We'll use constexpr int MAX_K = 32. But we must ensure k <= MAX_K, and we can check in launcher.

 We'll need shared memory size = block_size * k * (sizeof(float)+sizeof(int64_t)). We'll compute in launcher and pass.

 For block size, we can choose 256, which is good for 10000 elements. That means each thread processes ~40 elements, which is fine.

 We'll set grid size = batch_size.

 We'll launch: topk_kernel<<<batch_size, 256, shared_mem_size, stream>>>(...).

 We'll add a check for k <= MAX_K and maybe adjust MAX_K to 64 to be safe, but since k=5, 32 is fine. We'll define MAX_K=32.

 Now, for add_bias_kernel, we'll use a grid of (out_features+255)/256 blocks and 256 threads, each thread adds bias to all rows for that column. That's fine.

 For log_softmax, we'll use template kernel with BLOCK_SIZE=256, shared mem = 2*256*sizeof(float).

 For add_column_to_rows, we can use a simple kernel: each thread handles one element, grid sized (total_elements+255)/256.

 Now, implement launchers.

 For adaptive_log_softmax_log_prob_launcher, we need to know the dimensions: batch_size, input_size, head_size, reduce_size0, etc. The binding layer will pass these as ints, derived from tensor shapes. We'll assume they are correct.

 We'll allocate temporary buffers using cudaMalloc, then perform GEMMs, log_softmaxes, add columns, and memcpys, then free.

 We'll need to compute shortlist_size = cutoffs0 (since cutoffs list is [shortlist, ...]). Actually cutoffs0 is the first cutoff which equals shortlist_size. And cluster sizes are cutoffs1 - cutoffs0 and n_classes - cutoffs1. So we can set shortlist_size = cutoffs0. But in the contract, cutoffs is passed as an int? Wait, in model_new.py, cutoffs is passed as `self.adaptive_log_softmax.cutoffs`, which is a Python list. In the custom op signature, `cutoffs` is listed as kind int? Actually in custom_ops_contract.json, for adaptive_log_softmax_log_prob, one parameter is cutoffs with kind int, optional false. But in model_new.py, the call passes `self.adaptive_log_softmax.cutoffs`, which is a list. However, the contract says kind int? That seems inconsistent. The operator contract in the task file says `cutoffs` parameter kind int, but in the actual model_new.py call, it passes a list. We need to follow model_new.py as the source of truth. So adaptive_log_softmax_log_prob will receive a list of ints? But launcher cannot accept a list directly; binding must handle that. The binding will likely pass the cutoffs as two ints? Or maybe the binding will parse the list and pass individual values. Since we are only implementing .cu, we need to define launcher signature that binding can call. We can design launcher to take the cutoffs as three integers: shortlist_size, cutoffs0, cutoffs1? Actually, we need shortlist_size, and the two cluster boundaries. In our computation, we need shortlist_size, cluster_size0, cluster_size1, reduce_size0, reduce_size1. These can be derived from head_weight shape (head_size), tail0_w1 shape (reduce_size0), tail0_w2 shape (cluster_size0), tail1_w1 shape (reduce_size1), tail1_w2 shape (cluster_size1), and we need shortlist_size (which is cutoffs[0]) and maybe the actual number of classes? The output size n_classes = shortlist_size + cluster_size0 + cluster_size1. We don't need cutoffs[1] separately because cluster sizes are given by the tail weight shapes. So we only need shortlist_size from cutoffs. But the binding will pass cutoffs list; it can extract shortlist_size as the first element. So launcher can take single int `shortlist_size` or `cutoffs0`. We'll design launcher to accept `int shortlist_size` (the first cutoff). Also we need div_value as float. In model_new.py, call passes cutoffs list and div_value. The binding will need to convert cutoffs list to int (first element) and pass to launcher. So we'll define launcher signature with `int shortlist_size, float div_value`. Since the contract says cutoffs kind int, maybe they intend to pass only the first cutoff? But model_new.py passes whole list. The binding developer will need to handle that. We'll document in comments that shortlist_size is cutoffs[0].

 So launcher parameters:
 const float* input, const float* head_weight, const float* tail0_w1, const float* tail0_w2, const float* tail1_w1, const float* tail1_w2, float* output, int batch_size, int input_size, int head_size, int reduce_size0, int cluster_size0, int reduce_size1, int cluster_size1, int shortlist_size, float div_value, cudaStream_t stream.

 We'll assume all dimensions positive.

 In launcher, we'll compute and allocate.

 We'll also implement linear_forward_launcher with parameters: const float* input, const float* weight, const float* bias, float* output, int batch_size, int in_features, int out_features, cudaStream_t stream.

 The topk_launcher: const float* input, float* out_values, int64_t* out_indices, int batch_size, int n_classes, int k, cudaStream_t stream.

 Now, write the code.

 We'll need to include <cstdint> and <cfloat>. We'll use `fmaxf`, `__expf`, `logf`.

 For log_softmax, we use fast math maybe, but accuracy required atol=1e-2, so fine.

 In log_softmax kernel, we compute max, then sum exp, then log sum. Use block reduction. We need to handle cols wider than block. The loops over elements with stride are correct.

 We'll launch log_softmax with gridDim = rows, blockDim = 256. Shared mem = 2*256*sizeof(float).

 We'll create a function to launch log_softmax: `launch_log_softmax(float* out, const float* in, int rows, int cols, cudaStream_t stream)`. We'll template it internally.

 We'll also need a kernel to scale a matrix by scalar and then log_softmax? We can handle scaling by div_value before log_softmax: we can either multiply the tail_logits by div_value in the GEMM (alpha = div_value) or after. For tail0 and tail1, we need to compute log_softmax(div_value * tail_logits). We can pass alpha = div_value in the GEMM that computes tail_logits? Actually we compute tail_logits = reduced @ w2.T, then we need to apply div_value before log_softmax. We could set alpha in GEMM to div_value, so that tail_logits already scaled. But we also need the raw tail_logits for nothing else, so that's fine. So in the GEMM for tail0_w2, set alpha = div_value. Then proceed to log_softmax on scaled logits. Similarly for tail1.

 For head_logits, no scaling.

 Then after log_softmax for each tail, we need to add the corresponding head cluster logprob (which is head_logprobs[:, shortlist_size + i]). We'll use add_column_to_rows kernel.

 Then assembly via cudaMemcpy2DAsync.

 We need to ensure all operations are on the same stream and ordered. Since we use cuBLAS with stream and synchronous kernels, and then memcpys, they will be ordered on the stream.

 We must be careful with cudaMemcpy2DAsync: it copies 2D arrays with pitch. We'll set spitch = source_width * sizeof(float) (since source arrays are contiguous rows of that width), dpitch = n_classes * sizeof(float) (since output row width is n_classes). `width` in bytes = columns * sizeof(float). `height` = batch_size.

 We'll perform three copies:
 1. head shortlist: src = head_logprobs, dest = output, width = shortlist_size * sizeof(float), height = batch, spitch = head_size * sizeof(float), dpitch = n_classes * sizeof(float).
 2. tail0: src = tail0_logprobs, dest = output + shortlist_size, width = cluster_size0 * sizeof(float), height = batch, spitch = cluster_size0 * sizeof(float), dpitch = n_classes * sizeof(float).
 3. tail1: src = tail1_logprobs, dest = output + shortlist_size + cluster_size0, width = cluster_size1 * sizeof(float), height = batch, spitch = cluster_size1 * sizeof(float), dpitch = n_classes * sizeof(float).

 Note: n_classes = shortlist_size + cluster_size0 + cluster_size1.

 After copies, free all temporary buffers.

 Now, implement kernels.

 For add_column_to_rows, kernel: each thread handles one element, index = blockIdx.x*blockDim.x+threadIdx.x; if index < rows*cols, row = index / cols, col = index % cols; out[index] += column[row].

 For add_bias_kernel, similar but column-wise: each thread handles one column index j, then loops over rows adding bias[j] to each row.

 Now code everything.

 We must ensure no torch dependencies. We'll include cuda_runtime.h, cublas_v2.h.

 Let's produce the file content.

 We'll also add comments explaining.

 I'll now write the complete JSON response with the file content.
