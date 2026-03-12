//-----------------------------------------------------------------------------------------------
// AWQ (Activation-aware Weight Quantization) GPT-2 Model - Type Definitions and Memory Utilities
// 
// Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita
// ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef __GPT2_AWQ_TYPES_CUH__
#define __GPT2_AWQ_TYPES_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "utils.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// AWQ GPT-2 Configuration and Type Definitions

typedef struct {
    int max_seq_len;
    int vocab_size;
    int padded_vocab_size;  // padded to multiple of 128
    int num_layers;
    int num_heads;
    int channels;
    int quant_group_size; // should be 128
} AWQConfig;

// Non-quantized parameters
#define NUM_NON_QUANT_PARAMETER_TENSORS 12
typedef struct {
    // Embeddings
    float* wte;             // (Vp, C) - token embeddings
    float* wpe;             // (maxT, C) - position embeddings
    // Layer Norms
    float* ln1w;            // (L, C) - layer norm 1 weights
    float* ln1b;            // (L, C) - layer norm 1 biases
    float* ln2w;            // (L, C) - layer norm 2 weights
    float* ln2b;            // (L, C) - layer norm 2 biases
    float* lnfw;            // (C) - final layer norm weights
    float* lnfb;            // (C) - final layer norm biases
    // Other biases
    float* qkvb;            // (L, 3*C) - QKV biases
    float* attprojb;        // (L, C) - attention projection biases
    float* fcb;             // (L, 4*C) - FC biases
    float* fcprojb;         // (L, C) - FC projection biases
} NonQuantParamTensors;

// Quantized parameters
#define NUM_QUANT_PARAMETER_TENSORS 4
typedef struct {
    // Quantized weights (uint32 packed, 4 uint8 per uint32)
    uint32_t* qkvw_packed;      // (L, 3*C, C/4)
    uint32_t* attprojw_packed;  // (L, C, C/4)
    uint32_t* fcw_packed;       // (L, 4*C, C/4)
    uint32_t* fcprojw_packed;   // (L, C, C)
} QuantParamTensors;

// Dequantization parameters
typedef struct { // Quantization factors
    float* qkvw_q_factors;          // (L, 3*C, num_groups_C)
    float* attprojw_q_factors;      // (L, C, num_groups_C)
    float* fcw_q_factors;           // (L, 4*C, num_groups_C)
    float* fcprojw_q_factors;       // (L, C, num_groups_4C)
} DequantFactors;
typedef struct { // Zero weights
    uint8_t* qkvw_zero_points;      // (L, 3*C, num_groups_C)
    uint8_t* attprojw_zero_points;  // (L, C, num_groups_C)
    uint8_t* fcw_zero_points;       // (L, 4*C, num_groups_C)
    uint8_t* fcprojw_zero_points;   // (L, C, num_groups_4C)
} ZeroPoints;

#define NUM_ACTIVATION_TENSORS 21
// Same as gpt2.cuh
typedef struct {
    float* encoded; // (B, T, C)
    float* ln1; // (L, B, T, C)
    float* ln1_mean; // (L, B, T)
    float* ln1_rstd; // (L, B, T)
    float* atty; // (L, B, T, C)
    float* att; // (L, B, NH, T, T)
    float* attproj; // (L, B, T, C)
    float* residual2; // (L, B, T, C)
    float* ln2; // (L, B, T, C)
    float* ln2_mean; // (L, B, T)
    float* ln2_rstd; // (L, B, T)
    float* fch; // (L, B, T, 4*C)
    float* fch_gelu; // (L, B, T, 4*C)
    float* fcproj; // (L, B, T, C)
    float* residual3; // (L, B, T, C)
    float* lnf; // (B, T, C)
    float* lnf_mean; // (B, T)
    float* lnf_rstd; // (B, T)
    float* losses; // (B, T)
    float* qkvr; // (L, B, T, 3*C)
    // This buffer will store the logits
    // during the processing of transformer blocks, we will also use this as a
    // general scratchpad buffer. Allocation is made large enough to hold (B, T, 3C),
    // (B, NH, T, T), and (B, T, V) shaped tensors.
    float* output;
} AWQActivationTensors;

typedef struct {
    AWQConfig config;
    // Parameters - using separate structs for modularity
    NonQuantParamTensors nq_params;
    float* nq_memory;
    QuantParamTensors q_params;
    uint32_t* q_memory;
    DequantFactors dq_factors;
    float* dq_memory;
    ZeroPoints zp;
    uint8_t* zp_memory;
    size_t num_parameters;
    size_t num_parameters_quantized;
    // AWQ per-channel scales (optional - for fused operations)
    float* scales_qkv;           // (L, C) - scales for ln1 -> qkv
    float* scales_attproj;       // (L, C) - scales for atty -> attproj
    float* scales_fc;            // (L, C) - scales for ln2 -> fc
    float* scales_fcproj;        // (L, 4*C) - scales for fch_gelu -> fcproj
    // cuBLAS temporary buffers (pre-allocated to avoid repeated mallocs)
    float* d_weight_temp;        // Temp buffer for dequantized weights: max(3*C*C, 4*C*C, C*4*C)
    float* d_inp_scaled_temp;    // Temp buffer for scaled input: B*T*max(C, 4*C)
    // Activations
    AWQActivationTensors acts;
    float* acts_memory;
    size_t num_activations;
    // Runtime state
    int batch_size;
    int seq_len;
    int* inputs;
    float mean_loss;
    float* cpu_losses;
} GPT2_AWQ;

// ----------------------------------------------------------------------------
// Memory Allocation Utilities

// Helper function to accumulate total size from array
size_t accumulate_sizes(const size_t* sizes, int n) {
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        total += sizes[i];
    }
    return total;
}

// Generic memory allocation function with support for different data types
template<typename T>
T* malloc_and_point_generic(T** targets[], const size_t* sizes, int n,
                            size_t total_elements, bool on_device = true, bool zero_memory = false) {
    // Allocate memory
    T* memory;
    if (on_device) {
        cudaCheck(cudaMalloc((void**)&memory, total_elements * sizeof(T)));
        if (zero_memory) {
            cudaCheck(cudaMemset(memory, 0, total_elements * sizeof(T)));
        }
    } else {
        memory = (T*)mallocCheck(total_elements * sizeof(T));
        if (zero_memory) {
            memset(memory, 0, total_elements * sizeof(T));
        }
    }
    
    // Assign pointers to each tensor
    T* memory_iterator = memory;
    for (size_t i = 0; i < n; i++) {
        *(targets[i]) = memory_iterator;
        memory_iterator += sizes[i];
    }
    
    return memory;
}

// Fill in size arrays for each tensor type
void fill_in_nq_param_sizes(size_t* sizes, AWQConfig config) {
    int L = config.num_layers;
    int C = config.channels;
    int Vp = config.padded_vocab_size;
    int maxT = config.max_seq_len;
    
    sizes[0] = Vp * C;      // wte
    sizes[1] = maxT * C;    // wpe
    sizes[2] = L * C;       // ln1w
    sizes[3] = L * C;       // ln1b
    sizes[4] = L * C;       // ln2w
    sizes[5] = L * C;       // ln2b
    sizes[6] = C;           // lnfw
    sizes[7] = C;           // lnfb
    sizes[8] = L * 3 * C;   // qkvb
    sizes[9] = L * C;       // attprojb
    sizes[10] = L * 4 * C;  // fcb
    sizes[11] = L * C;      // fcprojb
}

void fill_in_q_param_sizes(size_t* sizes, AWQConfig config) {
    int L = config.num_layers;
    int C = config.channels;
    
    sizes[0] = L * 3 * C * (C / 4);    // qkvw_packed
    sizes[1] = L * C * (C / 4);        // attprojw_packed
    sizes[2] = L * 4 * C * (C / 4);    // fcw_packed
    sizes[3] = L * C * (4 * C / 4);    // fcprojw_packed
}

void fill_in_dq_factor_sizes(size_t* sizes, AWQConfig config, int group_size = 128) {
    int L = config.num_layers;
    int C = config.channels;
    int num_groups_C = C / group_size;
    int num_groups_4C = (4 * C) / group_size;
    
    sizes[0] = L * 3 * C * num_groups_C;   // qkvw_q_factors
    sizes[1] = L * C * num_groups_C;       // attprojw_q_factors
    sizes[2] = L * 4 * C * num_groups_C;   // fcw_q_factors
    sizes[3] = L * C * num_groups_4C;      // fcprojw_q_factors
}

void fill_in_zero_point_sizes(size_t* sizes, AWQConfig config, int group_size = 128) {
    // Zero points have same sizes as dequant factors
    fill_in_dq_factor_sizes(sizes, config, group_size);
}

void fill_in_activation_sizes(size_t* act_sizes, int B, int T, AWQConfig config) {
    size_t Vp = config.padded_vocab_size;
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    
    act_sizes[0] = B * T * C; // encoded
    act_sizes[1] = L * B * T * C; // ln1
    act_sizes[2] = 1; // ln1_mean, reserved for future use
    act_sizes[3] = 1; // ln1_rstd, reserved for future use
    act_sizes[4] = L * B * T * C; // atty
    act_sizes[5] = L * B * NH * T * T; // att
    act_sizes[6] = L * B * T * C; // attproj
    act_sizes[7] = L * B * T * C; // residual2
    act_sizes[8] = L * B * T * C; // ln2
    act_sizes[9] = 1; // ln2_mean, reserved for future use
    act_sizes[10] = 1; // ln2_rstd, reserved for future use
    act_sizes[11] = L * B * T * 4*C; // fch
    act_sizes[12] = L * B * T * 4*C; // fch_gelu
    act_sizes[13] = L * B * T * C; // fcproj
    act_sizes[14] = L * B * T * C; // residual3
    act_sizes[15] = B * T * C; // lnf
    act_sizes[16] = 1; // lnf_mean, reserved for future use
    act_sizes[17] = 1; // lnf_rstd, reserved for future use
    act_sizes[18] = B * T; // losses
    act_sizes[19] = L * B * T * 3*C; // qkvr
    act_sizes[20] = B * T * max(3*C, max(NH*T, Vp)); // output / scratch
}

// Wrapper functions for each tensor type
float* malloc_and_point_nq_params(NonQuantParamTensors* params, const size_t* sizes, size_t total, bool on_device = true) {
    float** ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b,
        &params->ln2w, &params->ln2b, &params->lnfw, &params->lnfb,
        &params->qkvb, &params->attprojb, &params->fcb, &params->fcprojb
    };
    return malloc_and_point_generic<float>(ptrs, sizes, NUM_NON_QUANT_PARAMETER_TENSORS, total, on_device, false);
}

uint32_t* malloc_and_point_q_params(QuantParamTensors* params, const size_t* sizes, size_t total, bool on_device = true) {
    uint32_t** ptrs[] = {
        &params->qkvw_packed, &params->attprojw_packed,
        &params->fcw_packed, &params->fcprojw_packed
    };
    return malloc_and_point_generic<uint32_t>(ptrs, sizes, NUM_QUANT_PARAMETER_TENSORS, total, on_device, false);
}

float* malloc_and_point_dq_factors(DequantFactors* factors, const size_t* sizes, size_t total, bool on_device = true) {
    float** ptrs[] = {
        &factors->qkvw_q_factors, &factors->attprojw_q_factors,
        &factors->fcw_q_factors, &factors->fcprojw_q_factors
    };
    return malloc_and_point_generic<float>(ptrs, sizes, NUM_QUANT_PARAMETER_TENSORS, total, on_device, false);
}

uint8_t* malloc_and_point_zero_points(ZeroPoints* zp, const size_t* sizes, size_t total, bool on_device = true) {
    uint8_t** ptrs[] = {
        &zp->qkvw_zero_points, &zp->attprojw_zero_points,
        &zp->fcw_zero_points, &zp->fcprojw_zero_points
    };
    return malloc_and_point_generic<uint8_t>(ptrs, sizes, NUM_QUANT_PARAMETER_TENSORS, total, on_device, false);
}

float* malloc_and_point_activations(AWQActivationTensors* acts, const size_t* act_sizes, size_t total) {
    float** ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->atty,
        &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->losses, &acts->qkvr, &acts->output
    };
    // Activations are always on GPU and always zeroed
    return malloc_and_point_generic<float>(ptrs, act_sizes, NUM_ACTIVATION_TENSORS, total, true, true);
}

#endif // __GPT2_AWQ_TYPES_CUH__
