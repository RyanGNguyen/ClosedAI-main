//-----------------------------------------------------------------------------------------------
// AWQ (Activation-aware Weight Quantization) GPT-2 Model
// 
// Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita
// ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef __GPT2_AWQ_CUH__
#define __GPT2_AWQ_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils/utils.h"
#include "utils/cuda_utils.cuh"
#include "utils/gpt2_awq_types.cuh"

// Standard GPT2 kernels
#include "kernels/encoder.cuh"
#include "kernels/attention.cuh"
#include "kernels/gelu.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/residual.cuh"
#include "kernels/matmul.cuh"

// AWQ kernels
#include "kernels_awq/dequantize.cuh"

// ----------------------------------------------------------------------------
// Core AWQ Model Functions

// Load AWQ activation scales from awq_optimal_scales.bin
void load_awq_scales(GPT2_AWQ *model, const char* scales_path) {
    FILE *scales_file = fopenCheck(scales_path, "rb");
    
    // Read header
    int file_L, file_C;
    freadCheck(&file_L, sizeof(int), 1, scales_file);
    freadCheck(&file_C, sizeof(int), 1, scales_file);
    
    if (file_L != model->config.num_layers || file_C != model->config.channels) {
        fprintf(stderr, "GPT2_AWQ - Scale file dimensions don't match model\n");
        fprintf(stderr, "  File: L=%d, C=%d | Model: L=%d, C=%d\n", 
                file_L, file_C, model->config.num_layers, model->config.channels);
        exit(EXIT_FAILURE);
    }
    
    int L = model->config.num_layers;
    int C = model->config.channels;
    
    // Total size: L*C*7 (layout: qkv, attproj, fc, fcproj with sizes L*C, L*C, L*C, L*4*C)
    size_t total_size = L * C * 7;
    float* h_scales = (float*)mallocCheck(total_size * sizeof(float));
    freadCheck(h_scales, sizeof(float), total_size, scales_file);
    fcloseCheck(scales_file);
    
    // Allocate GPU memory for each scale buffer
    cudaCheck(cudaMalloc(&model->scales_qkv, L * C * sizeof(float)));
    cudaCheck(cudaMalloc(&model->scales_attproj, L * C * sizeof(float)));
    cudaCheck(cudaMalloc(&model->scales_fc, L * C * sizeof(float)));
    cudaCheck(cudaMalloc(&model->scales_fcproj, L * 4 * C * sizeof(float)));
    
    // Copy scales to GPU (extract from L*C*7 layout)
    cudaCheck(cudaMemcpy(model->scales_qkv, h_scales, L * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(model->scales_attproj, h_scales + L * C, L * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(model->scales_fc, h_scales + 2 * L * C, L * C * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(model->scales_fcproj, h_scales + 3 * L * C, L * 4 * C * sizeof(float), cudaMemcpyHostToDevice));
    
    free(h_scales);
    
    printf("GPT2_AWQ - Loaded activation scales from %s\n", scales_path);
}

// Build AWQ model from checkpoint file
void gpt2_awq_build_from_checkpoint(GPT2_AWQ *model, const char* checkpoint_path) {
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    
    if (model_header[0] != 20250101) {
        fprintf(stderr, "GPT2_AWQ - Bad magic number (expected AWQ checkpoint)\n");
        exit(EXIT_FAILURE);
    }
    if (model_header[1] != 1) {
        fprintf(stderr, "GPT2_AWQ - Bad version in model file\n");
        exit(EXIT_FAILURE);
    }
    
    // Read hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];
    model->config.quant_group_size = model_header[8];
    
    int L = model->config.num_layers;
    int C = model->config.channels;
    int Vp = model->config.padded_vocab_size;
    int maxT = model->config.max_seq_len;
    int group_size = model->config.quant_group_size;
    int num_groups_C = C / group_size;
    int num_groups_4C = (4 * C) / group_size;
    
    printf("GPT2_AWQ - Loading AWQ model from %s\n", checkpoint_path);
    printf("  L=%d, C=%d, Vp=%d, maxT=%d\n", L, C, Vp, maxT);
    printf("  Group size=%d, num_groups_C=%d, num_groups_4C=%d\n", 
           group_size, num_groups_C, num_groups_4C);
    
    // Use the modular allocation functions
    size_t nq_sizes[NUM_NON_QUANT_PARAMETER_TENSORS];
    size_t q_sizes[NUM_QUANT_PARAMETER_TENSORS];
    size_t dq_sizes[NUM_QUANT_PARAMETER_TENSORS];
    size_t zp_sizes[NUM_QUANT_PARAMETER_TENSORS];
    
    fill_in_nq_param_sizes(nq_sizes, model->config);
    fill_in_q_param_sizes(q_sizes, model->config);
    fill_in_dq_factor_sizes(dq_sizes, model->config, group_size);
    fill_in_zero_point_sizes(zp_sizes, model->config, group_size);
    
    // Calculate totals using helper function
    size_t total_nq = accumulate_sizes(nq_sizes, NUM_NON_QUANT_PARAMETER_TENSORS);
    size_t total_dq = accumulate_sizes(dq_sizes, NUM_QUANT_PARAMETER_TENSORS);
    size_t total_zp = accumulate_sizes(zp_sizes, NUM_QUANT_PARAMETER_TENSORS);
    size_t total_q = accumulate_sizes(q_sizes, NUM_QUANT_PARAMETER_TENSORS);
    
    // Allocate CPU buffers for reading from file
    NonQuantParamTensors nq_params_cpu;
    QuantParamTensors q_params_cpu;
    DequantFactors dq_factors_cpu;
    ZeroPoints zp_cpu;
    
    float* nq_memory_cpu = malloc_and_point_nq_params(&nq_params_cpu, nq_sizes, total_nq, false);
    uint32_t* q_memory_cpu = malloc_and_point_q_params(&q_params_cpu, q_sizes, total_q, false);
    float* dq_memory_cpu = malloc_and_point_dq_factors(&dq_factors_cpu, dq_sizes, total_dq, false);
    uint8_t* zp_memory_cpu = malloc_and_point_zero_points(&zp_cpu, zp_sizes, total_zp, false);
    
    // Read from file to CPU buffers
    freadCheck(nq_memory_cpu, sizeof(float), total_nq, model_file);
    freadCheck(dq_memory_cpu, sizeof(float), total_dq, model_file);
    freadCheck(zp_memory_cpu, sizeof(uint8_t), total_zp, model_file);
    freadCheck(q_memory_cpu, sizeof(uint32_t), total_q, model_file);
    fcloseCheck(model_file);
    
    // Allocate GPU memory and copy from CPU
    model->nq_memory = malloc_and_point_nq_params(&model->nq_params, nq_sizes, total_nq, true);
    model->q_memory = malloc_and_point_q_params(&model->q_params, q_sizes, total_q, true);
    model->dq_memory = malloc_and_point_dq_factors(&model->dq_factors, dq_sizes, total_dq, true);
    model->zp_memory = malloc_and_point_zero_points(&model->zp, zp_sizes, total_zp, true);
    
    cudaCheck(cudaMemcpy(model->nq_memory, nq_memory_cpu, total_nq * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(model->dq_memory, dq_memory_cpu, total_dq * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(model->zp_memory, zp_memory_cpu, total_zp * sizeof(uint8_t), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(model->q_memory, q_memory_cpu, total_q * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    // Free CPU buffers
    free(nq_memory_cpu);
    free(q_memory_cpu);
    free(dq_memory_cpu);
    free(zp_memory_cpu);
    
    // Store sizes
    size_t total_fp32 = total_nq + total_dq;
    model->num_parameters = total_fp32 + (total_q * 4);  // uint32 = 4 uint8
    model->num_parameters_quantized = total_q * 4;  // Stored as 8-bit
    
    printf("GPT2_AWQ - Loaded %zu total parameters\n", model->num_parameters);
    printf("  FP32 parameters: %zu (%zu MiB)\n", total_fp32, (total_fp32 * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
    printf("  INT8 parameters: %zu (%zu MiB)\n", model->num_parameters_quantized, 
           (model->num_parameters_quantized * sizeof(uint8_t)) >> 20);
    printf("  Total size: %zu MiB\n", 
           ((total_fp32 * sizeof(float) + model->num_parameters_quantized) >> 20));
    
    // Other inits
    model->acts_memory = NULL;
    model->inputs = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f;
    model->scales_qkv = NULL;
    model->scales_attproj = NULL;
    model->scales_fc = NULL;
    model->scales_fcproj = NULL;
    model->d_weight_temp = NULL;
    model->d_inp_scaled_temp = NULL;
    
    // Load activation scales from awq_optimal_scales.bin
    // This file should be in the same directory as the checkpoint
    char scales_path[512];
    snprintf(scales_path, sizeof(scales_path), "awq_optimal_scales.bin");
    
    // Check if scales file exists
    FILE* test_scales = fopen(scales_path, "rb");
    if (test_scales != NULL) {
        fclose(test_scales);
        load_awq_scales(model, scales_path);
    } else {
        fprintf(stderr, "GPT2_AWQ - Warning: Activation scales file not found at %s\n", scales_path);
        fprintf(stderr, "            Forward pass will NOT apply activation scaling!\n");
    }
}

// AWQ forward pass with fused dequantization
// Note: This simplified version uses standard matmul with on-the-fly dequantization
// For production, use fused_dequant_scale_matmul from dequantize.cuh
void gpt2_awq_forward(GPT2_AWQ *model, int* inputs, int B, int T) {
    
    // Ensure the model was initialized
    if (model->nq_memory == NULL || model->q_memory == NULL || 
        model->dq_memory == NULL || model->zp_memory == NULL) {
        printf("GPT2_AWQ - Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // Validate inputs
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
    }

    // Allocate activations if needed
    if(model->acts_memory == NULL || B != model->batch_size || T != model->seq_len) {
        model->batch_size = B;
        model->seq_len = T;
        
        if (model->acts_memory != NULL) {
            cudaFree(model->acts_memory);
            cudaFree(model->inputs);
            if (model->d_weight_temp) cudaFree(model->d_weight_temp);
            if (model->d_inp_scaled_temp) cudaFree(model->d_inp_scaled_temp);
        }
        
        // Calculate activation sizes using helper function
        size_t act_sizes[NUM_ACTIVATION_TENSORS];
        fill_in_activation_sizes(act_sizes, B, T, model->config);
        size_t num_activations = accumulate_sizes(act_sizes, NUM_ACTIVATION_TENSORS);
        model->acts_memory = malloc_and_point_activations(&model->acts, act_sizes, num_activations);
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
        
        // Allocate cuBLAS temp buffers (pre-allocated to avoid repeated mallocs)
        // d_weight_temp: max(3*C*C, 4*C*C, C*4*C) = 4*C*C floats
        // d_inp_scaled_temp: B*T*max(C, 4*C) = B*T*4*C floats
        size_t weight_temp_size = 4 * C * C;
        size_t inp_scaled_temp_size = B * T * 4 * C;
        cudaCheck(cudaMalloc((void**)&model->d_weight_temp, weight_temp_size * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&model->d_inp_scaled_temp, inp_scaled_temp_size * sizeof(float)));
        
        model->num_activations = num_activations;
        printf("GPT2_AWQ - Allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20);
        printf("GPT2_AWQ - Allocated %zu MiB for cuBLAS temp buffers\n", 
               ((weight_temp_size + inp_scaled_temp_size) * sizeof(float)) >> 20);
    }

    // Copy inputs to device
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // Forward pass - use the correct parameter structs
    NonQuantParamTensors& nq_params = model->nq_params;
    QuantParamTensors& q_params = model->q_params;
    DequantFactors& dq_factors = model->dq_factors;
    ZeroPoints& zp = model->zp;
    AWQActivationTensors& acts = model->acts;
    float* residual;
    
    // Encoding: embedding lookup (standard, FP32)
    encoder_forward(acts.encoded, model->inputs, nq_params.wte, nq_params.wpe, B, T, C);

    // Check if activation scales are loaded
    bool use_fused_kernels = (model->scales_qkv != NULL && model->scales_attproj != NULL && 
                               model->scales_fc != NULL && model->scales_fcproj != NULL);
    
    if (!use_fused_kernels) {
        fprintf(stderr, "GPT2_AWQ - Warning: Running without activation scaling (scales not loaded)\n");
    }
    
    // Allocate temporary buffers for dequantized weights (only if not using fused kernels)
    int num_groups_C = C / 128;
    int num_groups_4C = (4 * C) / 128;
    
    float* d_qkvw_temp = NULL;
    float* d_attprojw_temp = NULL;
    float* d_fcw_temp = NULL;
    float* d_fcprojw_temp = NULL;
    
    if (!use_fused_kernels) {
        cudaCheck(cudaMalloc(&d_qkvw_temp, 3 * C * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_attprojw_temp, C * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_fcw_temp, 4 * C * C * sizeof(float)));
        cudaCheck(cudaMalloc(&d_fcprojw_temp, C * 4 * C * sizeof(float)));
    }

    for (int l = 0; l < L; l++) {
        residual = (l == 0) ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // Get pointers for this layer
        float* l_ln1w = nq_params.ln1w + l * C;
        float* l_ln1b = nq_params.ln1b + l * C;
        uint32_t* l_qkvw_packed = q_params.qkvw_packed + l * 3 * C * (C / 4);
        float* l_qkvw_qf = dq_factors.qkvw_q_factors + l * 3 * C * num_groups_C;
        uint8_t* l_qkvw_zp = zp.qkvw_zero_points + l * 3 * C * num_groups_C;
        float* l_qkvb = nq_params.qkvb + l * 3 * C;
        uint32_t* l_attprojw_packed = q_params.attprojw_packed + l * C * (C / 4);
        float* l_attprojw_qf = dq_factors.attprojw_q_factors + l * C * num_groups_C;
        uint8_t* l_attprojw_zp = zp.attprojw_zero_points + l * C * num_groups_C;
        float* l_attprojb = nq_params.attprojb + l * C;
        float* l_ln2w = nq_params.ln2w + l * C;
        float* l_ln2b = nq_params.ln2b + l * C;
        uint32_t* l_fcw_packed = q_params.fcw_packed + l * 4 * C * (C / 4);
        float* l_fcw_qf = dq_factors.fcw_q_factors + l * 4 * C * num_groups_C;
        uint8_t* l_fcw_zp = zp.fcw_zero_points + l * 4 * C * num_groups_C;
        float* l_fcb = nq_params.fcb + l * 4 * C;
        uint32_t* l_fcprojw_packed = q_params.fcprojw_packed + l * C * (4 * C / 4);
        float* l_fcprojw_qf = dq_factors.fcprojw_q_factors + l * C * num_groups_4C;
        uint8_t* l_fcprojw_zp = zp.fcprojw_zero_points + l * C * num_groups_4C;
        float* l_fcprojb = nq_params.fcprojb + l * C;

        // Get activation pointers for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean;
        float* l_ln1_rstd = acts.ln1_rstd;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean;
        float* l_ln2_rstd = acts.ln2_rstd;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        float* scratch = acts.output;

        // Get scale pointers for this layer (if available)
        float* l_scales_qkv = use_fused_kernels ? (model->scales_qkv + l * C) : NULL;
        float* l_scales_attproj = use_fused_kernels ? (model->scales_attproj + l * C) : NULL;
        float* l_scales_fc = use_fused_kernels ? (model->scales_fc + l * C) : NULL;
        float* l_scales_fcproj = use_fused_kernels ? (model->scales_fcproj + l * 4 * C) : NULL;

        // Layer norm 1
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);

        // QKV projection with cuBLAS-accelerated dequantization + activation scaling
        if (use_fused_kernels) {
            fused_dequant_scale_matmul_bias_cublas(scratch, l_ln1, l_qkvw_packed, l_scales_qkv,
                                                    l_qkvw_qf, l_qkvw_zp, l_qkvb,
                                                    model->d_weight_temp, model->d_inp_scaled_temp,
                                                    B, T, C, 3*C);
        } else {
            // Fallback: dequantize + matmul (without activation scaling)
            dequantize_weights(d_qkvw_temp, l_qkvw_packed, l_qkvw_qf, l_qkvw_zp, 3*C, C);
            matmul_forward(scratch, l_ln1, d_qkvw_temp, l_qkvb, B, T, C, 3*C);
        }
        
        // Attention
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        
        // Attention projection with cuBLAS-accelerated dequantization + activation scaling
        if (use_fused_kernels) {
            fused_dequant_scale_matmul_bias_cublas(l_attproj, l_atty, l_attprojw_packed, l_scales_attproj,
                                                    l_attprojw_qf, l_attprojw_zp, l_attprojb,
                                                    model->d_weight_temp, model->d_inp_scaled_temp,
                                                    B, T, C, C);
        } else {
            // Fallback: dequantize + matmul (without activation scaling)
            dequantize_weights(d_attprojw_temp, l_attprojw_packed, l_attprojw_qf, l_attprojw_zp, C, C);
            matmul_forward(l_attproj, l_atty, d_attprojw_temp, l_attprojb, B, T, C, C);
        }
        
        // Residual connection
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        
        // Layer norm 2
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        
        // FC layer with cuBLAS-accelerated dequantization + activation scaling
        if (use_fused_kernels) {
            fused_dequant_scale_matmul_bias_cublas(l_fch, l_ln2, l_fcw_packed, l_scales_fc,
                                                    l_fcw_qf, l_fcw_zp, l_fcb,
                                                    model->d_weight_temp, model->d_inp_scaled_temp,
                                                    B, T, C, 4*C);
        } else {
            // Fallback: dequantize + matmul (without activation scaling)
            dequantize_weights(d_fcw_temp, l_fcw_packed, l_fcw_qf, l_fcw_zp, 4*C, C);
            matmul_forward(l_fch, l_ln2, d_fcw_temp, l_fcb, B, T, C, 4*C);
        }
        
        // GELU activation
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        
        // FC projection with cuBLAS-accelerated dequantization + activation scaling
        if (use_fused_kernels) {
            fused_dequant_scale_matmul_bias_cublas(l_fcproj, l_fch_gelu, l_fcprojw_packed, l_scales_fcproj,
                                                    l_fcprojw_qf, l_fcprojw_zp, l_fcprojb,
                                                    model->d_weight_temp, model->d_inp_scaled_temp,
                                                    B, T, 4*C, C);
        } else {
            // Fallback: dequantize + matmul (without activation scaling)
            dequantize_weights(d_fcprojw_temp, l_fcprojw_packed, l_fcprojw_qf, l_fcprojw_zp, C, 4*C);
            matmul_forward(l_fcproj, l_fch_gelu, d_fcprojw_temp, l_fcprojb, B, T, 4*C, C);
        }
        
        // Residual connection
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    // Final layer norm
    residual = acts.residual3 + (L-1) * B * T * C;
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, nq_params.lnfw, nq_params.lnfb, B, T, C);
    
    // Final projection (wte is FP32, no quantization)
    matmul_forward(acts.output, acts.lnf, nq_params.wte, NULL, B, T, C, Vp);

    // Free temporary buffers (only allocated if not using fused kernels)
    if (!use_fused_kernels) {
        cudaCheck(cudaFree(d_qkvw_temp));
        cudaCheck(cudaFree(d_attprojw_temp));
        cudaCheck(cudaFree(d_fcw_temp));
        cudaCheck(cudaFree(d_fcprojw_temp));
    }

    model->mean_loss = -1.0f;
}

void gpt2_awq_free(GPT2_AWQ *model) {
    // Free parameter memory blocks (all on GPU)
    if (model->nq_memory) cudaCheck(cudaFree(model->nq_memory));
    if (model->q_memory) cudaCheck(cudaFree(model->q_memory));
    if (model->dq_memory) cudaCheck(cudaFree(model->dq_memory));
    if (model->zp_memory) cudaCheck(cudaFree(model->zp_memory));
    
    // Free activation and runtime memory
    if (model->acts_memory) cudaCheck(cudaFree(model->acts_memory));
    if (model->inputs) cudaCheck(cudaFree(model->inputs));
    if (model->cpu_losses) cudaCheck(cudaFreeHost(model->cpu_losses));
    
    // Free optional scale buffers
    if (model->scales_qkv) cudaCheck(cudaFree(model->scales_qkv));
    if (model->scales_attproj) cudaCheck(cudaFree(model->scales_attproj));
    if (model->scales_fc) cudaCheck(cudaFree(model->scales_fc));
    if (model->scales_fcproj) cudaCheck(cudaFree(model->scales_fcproj));
    
    // Free cuBLAS temp buffers
    if (model->d_weight_temp) cudaCheck(cudaFree(model->d_weight_temp));
    if (model->d_inp_scaled_temp) cudaCheck(cudaFree(model->d_inp_scaled_temp));
}

#define GPT2_EOT 50256
#endif // __GPT2_AWQ_CUH__
