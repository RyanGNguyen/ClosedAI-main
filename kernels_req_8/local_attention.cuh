#ifndef LOCAL_ATTENTION_FORWARD_CUH
#define LOCAL_ATTENTION_FORWARD_CUH

#include <cuda_runtime.h>
#include <float.h>
#include <assert.h>

#define WINDOW_SIZE 128
#define WARP_SIZE 32

// Reuse existing permute kernel
__global__ void permute_kernel(float *q, float *k, float *v, const float *inp, 
                               int B, int N, int NH, int d) {
    int b = blockIdx.x;
    int nh_idx = blockIdx.y;
    
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        for (int i = 0; i < d; i++) {
            int inp_idx = (b * N * 3 * NH * d) + (t * 3 * NH * d) + (nh_idx * d) + i;
            int out_idx = (b * NH * N * d) + (nh_idx * N * d) + (t * d) + i;
            q[out_idx] = inp[inp_idx];
            k[out_idx] = inp[inp_idx + NH * d];
            v[out_idx] = inp[inp_idx + 2 * NH * d];
        }
    }
}

// Reuse existing unpermute kernel
__global__ void unpermute_kernel(float *inp, float *out, int B, int N, int NH, int d) {
    int b = blockIdx.x;
    int nh_idx = blockIdx.y;
    
    for (int t = threadIdx.x; t < N; t += blockDim.x) {
        for (int i = 0; i < d; i++) {
            int inp_idx = (b * NH * N * d) + (nh_idx * N * d) + (t * d) + i;
            int out_idx = (b * N * NH * d) + (t * NH * d) + (nh_idx * d) + i;
            out[out_idx] = inp[inp_idx];
        }
    }
}

// Combined kernel for QK^T and softmax within sliding window
__global__ void local_attention_scores_kernel(float *out, const float *q, const float *k,
                                              int B, int T, int NH, int HS, float scale) {
    extern __shared__ float shared[];

    int b = blockIdx.z;
    int nh = blockIdx.y;
    int q_pos = blockIdx.x;

    if (b >= B || q_pos >= T) return;

    int tid = threadIdx.x;
    int block_size = blockDim.x;

    // Calculate window boundaries (matching CPU: k_start = max(0, q_t - WINDOW_SIZE))
    int window_start = max(0, q_pos - WINDOW_SIZE);
    int window_end = q_pos;
    int window_len = window_end - window_start + 1;

    const float *q_ptr = q + (b * NH * T * HS) + (nh * T * HS) + (q_pos * HS);
    const float *k_base = k + (b * NH * T * HS) + (nh * T * HS);
    float *out_ptr = out + (b * NH * T * T) + (nh * T * T) + (q_pos * T);

    float *reduction_scratch = shared + WINDOW_SIZE + 1;

    // Phase 1: Compute QK^T scores for the window
    float local_max = -FLT_MAX;
    for (int k_pos = window_start + tid; k_pos <= window_end; k_pos += block_size) {
        float score = 0.0f;
        const float *k_ptr = k_base + k_pos * HS;
        
        // Dot product
        for (int i = 0; i < HS; i++) {
            score += q_ptr[i] * k_ptr[i];
        }
        score *= scale;
        shared[k_pos - window_start] = score;
        local_max = fmaxf(local_max, score);
    }

    // Reduce to find global max - write local_max to shared memory first
    reduction_scratch[tid] = local_max;
    __syncthreads();

    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            reduction_scratch[tid] = fmaxf(reduction_scratch[tid], reduction_scratch[tid + s]);
        }
        __syncthreads();
    }

    float global_max = reduction_scratch[0];
    __syncthreads();

    // Phase 2: Compute exp and sum
    float local_sum = 0.0f;
    for (int k_pos = window_start + tid; k_pos <= window_end; k_pos += block_size) {
        float exp_val = expf(shared[k_pos - window_start] - global_max);
        shared[k_pos - window_start] = exp_val;
        local_sum += exp_val;
    }

    // Reduce to find sum - write local_sum to shared memory first
    reduction_scratch[tid] = local_sum;
    __syncthreads();

    for (int s = block_size >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            reduction_scratch[tid] += reduction_scratch[tid + s];
        }
        __syncthreads();
    }

    float inv_sum = 1.0f / reduction_scratch[0];
    __syncthreads();

    // Phase 3: Write normalized probabilities
    for (int i = tid; i < T; i += block_size) {
        if (i >= window_start && i <= window_end) {
            out_ptr[i] = shared[i - window_start] * inv_sum;
        } else {
            out_ptr[i] = 0.0f;
        }
    }
}

// Attention @ V matmul
__global__ void local_pv_matmul_kernel(float *out, const float *P, const float *V,
                                       int B, int T, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = B * NH * T * HS;
    
    if (idx >= total_elements) return;
    
    int b = idx / (NH * T * HS);
    int nh = (idx / (T * HS)) % NH;
    int t = (idx / HS) % T;
    int hs = idx % HS;
    
    // Window for this query position (matching CPU)
    int window_start = max(0, t - WINDOW_SIZE);
    int window_end = t;
    
    const float *P_row = P + (b * NH * T * T) + (nh * T * T) + (t * T);
    const float *V_base = V + (b * NH * T * HS) + (nh * T * HS);
    
    float acc = 0.0f;
    for (int i = window_start; i <= window_end; i++) {
        acc += P_row[i] * V_base[i * HS + hs];
    }
    
    out[idx] = acc;
}

// Main forward function for the test harness
void local_attention_forward(float* out, float* qkvr, float* inp,
                            int B, int T, int NH, int HS) {
    
    float *q = qkvr + 0 * B * T * NH * HS;
    float *k = qkvr + 1 * B * T * NH * HS;
    float *v = qkvr + 2 * B * T * NH * HS;
    
    // Allocate attention probabilities
    float *probs;
    size_t probs_size = B * NH * T * T * sizeof(float);
    cudaMalloc(&probs, probs_size);
    
    // Step 1: Permute input
    dim3 permute_grid(B, NH);
    dim3 permute_block(256);
    permute_kernel<<<permute_grid, permute_block>>>(q, k, v, inp, B, T, NH, HS);
    
    // Step 2: Compute attention scores and softmax
    float scale = 1.0f / sqrtf(static_cast<float>(HS));
    dim3 scores_grid(T, NH, B);
    dim3 scores_block(256);
    // Allocate shared memory for: scores (WINDOW_SIZE+1) + reduction scratch (256 threads)
    size_t shared_size = (WINDOW_SIZE + 1 + 256) * sizeof(float);
    local_attention_scores_kernel<<<scores_grid, scores_block, shared_size>>>(
        probs, q, k, B, T, NH, HS, scale);
    
    // Step 3: Attention @ V (using inp as temporary buffer)
    int threads = 256;
    int blocks = (B * NH * T * HS + threads - 1) / threads;
    local_pv_matmul_kernel<<<blocks, threads>>>(inp, probs, v, B, T, NH, HS);
    
    // Step 4: Unpermute output
    unpermute_kernel<<<permute_grid, permute_block>>>(inp, out, B, T, NH, HS);
    
    cudaFree(probs);
}

#endif // __ATTENTION_CUH__