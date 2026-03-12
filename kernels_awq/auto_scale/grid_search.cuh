#ifndef __GRID_SEARCH_KERNEL_CUH__
#define __GRID_SEARCH_KERNEL_CUH__

#include "../../utils/cuda_utils.cuh"
#include "../../utils/gpt2_types.cuh"
#include "../quantize.cuh"
#include "../scale_tensors.cuh"
#include "normalize_scales.cuh"
#include "mse_loss.cuh"
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

// Forward declaration of apply_optimal_scales_to_model (defined later in this file)
void apply_optimal_scales_to_model(const float* d_optimal_scales, GPT2* model);

// =============================================
// GRID SEARCH FOR OPTIMAL ALPHA
// =============================================

// Matrix type indices for the 4 weight matrices we optimize
enum MatrixType {
    MATRIX_QKV = 0,      // ln1 -> QKV projection
    MATRIX_ATTPROJ = 1,  // atty -> Attention projection
    MATRIX_FC1 = 2,      // ln2 -> FC1
    MATRIX_FC2 = 3       // fch_gelu -> FC2
};

// Structure to hold grid search configuration
struct GridSearchConfig {
    int L;                      // Number of layers
    int C;                      // Number of channels
    int B;                      // Batch size
    int T;                      // Sequence length
    int num_alpha_values;       // Number of alpha values to test
    const float* alpha_values;  // Array of alpha values [0.0, 0.1, ..., 1.0]
};

// Helper function to extract channel averages for a specific matrix type
// Channel averages layout: [ln1 (L*C), atty (L*C), ln2 (L*C), fch_gelu (L*4*C)]
void extract_channel_avgs_for_matrix(
    float* d_extracted,              // Output: (L, IC) where IC is C or 4*C
    const float* d_channel_avgs,     // Input: full channel averages (L*C*7)
    MatrixType matrix_type,
    int L, int C)
{
    size_t offset = 0;
    size_t size = 0;
    
    switch (matrix_type) {
        case MATRIX_QKV:
            offset = 0;           // ln1
            size = L * C;
            break;
        case MATRIX_ATTPROJ:
            offset = L * C;       // atty
            size = L * C;
            break;
        case MATRIX_FC1:
            offset = 2 * L * C;   // ln2
            size = L * C;
            break;
        case MATRIX_FC2:
            offset = 3 * L * C;   // fch_gelu
            size = L * 4 * C;
            break;
    }
    
    cudaCheck(cudaMemcpy(d_extracted, d_channel_avgs + offset, 
                         size * sizeof(float), cudaMemcpyDeviceToDevice));
}

// Helper function to get weight and activation pointers for a matrix type
void get_matrix_pointers(
    float** weights,        // Output: (L, OC, IC) - pointer to weight matrix
    float** activations,    // Output: (L, B, T, IC) - pointer to activation matrix
    float** outputs,        // Output: (L, B, T, OC) - pointer to output matrix
    int* OC,                // Output: output channels
    int* IC,                // Output: input channels
    GPT2* model,
    MatrixType matrix_type)
{
    int C = model->config.channels;
    
    switch (matrix_type) {
        case MATRIX_QKV:
            *weights = model->params.qkvw;      // (L, 3*C, C)
            *activations = model->acts.ln1;     // (L, B, T, C)
            *outputs = model->acts.qkvr;        // (L, B, T, 3*C)
            *OC = 3 * C;
            *IC = C;
            break;
        case MATRIX_ATTPROJ:
            *weights = model->params.attprojw;  // (L, C, C)
            *activations = model->acts.atty;    // (L, B, T, C)
            *outputs = model->acts.attproj;     // (L, B, T, C)
            *OC = C;
            *IC = C;
            break;
        case MATRIX_FC1:
            *weights = model->params.fcw;       // (L, 4*C, C)
            *activations = model->acts.ln2;     // (L, B, T, C)
            *outputs = model->acts.fch;         // (L, B, T, 4*C)
            *OC = 4 * C;
            *IC = C;
            break;
        case MATRIX_FC2:
            *weights = model->params.fcprojw;   // (L, C, 4*C)
            *activations = model->acts.fch_gelu; // (L, B, T, 4*C)
            *outputs = model->acts.fcproj;      // (L, B, T, C)
            *OC = C;
            *IC = 4 * C;
            break;
    }
}

// Main grid search function for a single matrix type
// Finds optimal alpha per layer by minimizing MSE
void grid_search_matrix(
    float* h_best_alphas,           // Output: (L,) - best alpha per layer
    float* d_optimal_scales,        // Output: (L, IC) - optimal scales per layer
    const float* d_channel_avgs,    // Input: channel averages for this matrix
    GPT2* model,
    MatrixType matrix_type,
    int* batched_tokens,            // Input: (num_batches, B, T) - batched calibration tokens
    int num_batches,                // Number of batches to process
    const GridSearchConfig& config)
{
    int L = config.L;
    int B = config.B;
    int T = config.T;
    
    // Get matrix-specific parameters
    float *d_weights, *d_activations, *d_outputs;
    int OC, IC;
    get_matrix_pointers(&d_weights, &d_activations, &d_outputs, 
                        &OC, &IC, model, matrix_type);
    
    printf("Grid search for matrix type %d: OC=%d, IC=%d\n", matrix_type, OC, IC);
    
    // Allocate working memory
    float *d_scales;               // (L, IC) - current test scales
    float *d_weights_scaled;       // (L, OC, IC) - scaled weights
    float *d_weights_pq;           // (L, OC, IC) - pseudo-quantized weights
    float *d_activations_scaled;   // (L, B, T, IC) - scaled activations
    float *d_baseline_outputs;     // (L, B, T, OC) - baseline outputs
    float *d_candidate_outputs;    // (L, B, T, OC) - candidate outputs
    float *d_q_factors;            // Quantization factors
    uint8_t *d_zero_points;        // Zero points
    
    size_t act_size = L * B * T * IC * sizeof(float);
    size_t out_size = L * B * T * OC * sizeof(float);
    size_t weight_size = L * OC * IC * sizeof(float);
    size_t scale_size = L * IC * sizeof(float);
    
    cudaCheck(cudaMalloc(&d_scales, scale_size));
    cudaCheck(cudaMalloc(&d_weights_scaled, weight_size));
    cudaCheck(cudaMalloc(&d_weights_pq, weight_size));
    cudaCheck(cudaMalloc(&d_activations_scaled, act_size));
    cudaCheck(cudaMalloc(&d_baseline_outputs, out_size));
    cudaCheck(cudaMalloc(&d_candidate_outputs, out_size));
    
    int num_groups = CEIL_DIV(IC, GROUP_SIZE);
    cudaCheck(cudaMalloc(&d_q_factors, L * OC * num_groups * sizeof(float)));
    cudaCheck(cudaMalloc(&d_zero_points, L * OC * num_groups * sizeof(uint8_t)));
    
    // Track best alpha per layer
    float* h_best_mse = (float*)malloc(L * sizeof(float));
    for (int l = 0; l < L; l++) {
        h_best_mse[l] = FLT_MAX;
        h_best_alphas[l] = 0.0f;
    }
    
    // Grid search over alpha values
    for (int alpha_idx = 0; alpha_idx < config.num_alpha_values; alpha_idx++) {
        float alpha = config.alpha_values[alpha_idx];
        printf("  Testing alpha = %.2f\n", alpha);
        
        // Compute and normalize scales: s = (channel_avgs^alpha) / sqrt(max*min)
        cudaCheck(cudaMemcpy(d_scales, d_channel_avgs, scale_size, cudaMemcpyDeviceToDevice));
        normalize_scales(d_scales, L, IC, alpha);  // Does both power and normalization
        
        // Accumulate MSE across batches
        float* h_total_mse = (float*)calloc(L, sizeof(float));
        
        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            // Get pointer to this batch's tokens (B, T)
            int* batch_tokens = batched_tokens + batch_idx * B * T;
            
            // Run baseline forward pass to get Y_baseline
            gpt2_forward(model, batch_tokens, B, T);
            
            // Save baseline outputs (copy from model activations)
            cudaCheck(cudaMemcpy(d_baseline_outputs, d_outputs, out_size, cudaMemcpyDeviceToDevice));
            
            // Scale weights: W_scaled = W * diag(s) - ALL LAYERS IN PARALLEL
            scale_channels(d_weights_scaled, d_weights, d_scales, L, OC, IC);
            
            // Compute quantization parameters on scaled weights - ALL LAYERS IN PARALLEL
            calc_quant_param(d_q_factors, d_zero_points, d_weights_scaled, L, OC, IC);
            
            // Pseudo-quantize scaled weights - ALL LAYERS IN PARALLEL
            pseudo_quantize_weights(d_weights_pq, d_weights_scaled, 
                                           d_q_factors, d_zero_points, L, OC, IC);
            
            // Scale activations: X_scaled = X / diag(s) - ALL LAYERS IN PARALLEL
            scale_channels(d_activations_scaled, d_activations, d_scales, L, B * T, IC);
            
            // Compute Y_candidate = W_pq @ X_scaled using BATCHED cuBLAS
            // All L layers computed in one cuBLAS call
            float alpha_gemm = 1.0f;
            float beta_gemm = 0.0f;
            cublasCheck(cublasSgemmStridedBatched(cublas_handle,
                CUBLAS_OP_N,                // W is already transposed in memory
                CUBLAS_OP_N,                // X is not transposed
                OC,                         // M: rows of C
                B * T,                      // N: cols of C
                IC,                         // K: shared dimension
                &alpha_gemm,
                d_weights_pq, OC, OC * IC,  // A: (OC, IC) with lda=OC, stride between layers
                d_activations_scaled, IC, B * T * IC,  // B: (IC, B*T) with ldb=IC, stride
                &beta_gemm,
                d_candidate_outputs, OC, B * T * OC,   // C: (OC, B*T) with ldc=OC, stride
                L));                        // Batch count = L layers
            cudaCheck(cudaGetLastError());
            
            // Allocate device memory for per-layer MSE results
            float* d_mse_per_layer;
            cudaCheck(cudaMalloc(&d_mse_per_layer, L * sizeof(float)));
            
            // Compute MSE for ALL LAYERS IN PARALLEL
            mse_loss(d_mse_per_layer, d_candidate_outputs, d_baseline_outputs, 
                            L, B * T * OC);
            
            // Copy all L MSE values to host at once
            float* h_mse_batch = (float*)malloc(L * sizeof(float));
            cudaCheck(cudaMemcpy(h_mse_batch, d_mse_per_layer, L * sizeof(float), 
                                cudaMemcpyDeviceToHost));
            
            // Accumulate MSE for each layer
            for (int l = 0; l < L; l++) {
                h_total_mse[l] += h_mse_batch[l];
            }
            
            // Cleanup
            free(h_mse_batch);
            cudaCheck(cudaFree(d_mse_per_layer));
        }
        
        // Average MSE across samples and update best alpha per layer
        for (int l = 0; l < L; l++) {
            float avg_mse = h_total_mse[l] / num_batches;
            if (avg_mse < h_best_mse[l]) {
                h_best_mse[l] = avg_mse;
                h_best_alphas[l] = alpha;
            }
        }
        
        free(h_total_mse);
    }
    
    // Compute optimal scales using best alphas
    for (int l = 0; l < L; l++) {
        printf("  Layer %d: best alpha = %.2f (MSE = %.6f)\n", 
               l, h_best_alphas[l], h_best_mse[l]);
        
        // Extract this layer's channel averages
        float* d_layer_avgs = d_scales;  // Reuse buffer
        cudaCheck(cudaMemcpy(d_layer_avgs, 
                            d_channel_avgs + l * IC, 
                            IC * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
        
        // Apply best alpha and normalize in one call
        normalize_scales(d_layer_avgs, 1, IC, h_best_alphas[l]);
        
        // Copy to output
        cudaCheck(cudaMemcpy(d_optimal_scales + l * IC, 
                            d_layer_avgs, 
                            IC * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
    }
    
    // Cleanup
    cudaCheck(cudaFree(d_scales));
    cudaCheck(cudaFree(d_weights_scaled));
    cudaCheck(cudaFree(d_weights_pq));
    cudaCheck(cudaFree(d_activations_scaled));
    cudaCheck(cudaFree(d_baseline_outputs));
    cudaCheck(cudaFree(d_candidate_outputs));
    cudaCheck(cudaFree(d_q_factors));
    cudaCheck(cudaFree(d_zero_points));
    free(h_best_mse);
}

// Main entry point for grid search optimization
// Finds optimal alpha and scales for all 4 weight matrices in all layers
void grid_search_optimal_alpha(
    float* h_optimal_alphas,        // Output: (L, 4) - one alpha per (layer, matrix)
    float* d_optimal_scales,        // Output: (L*C*7) - optimized scale factors
    const float* d_channel_avgs,    // Input: (L*C*7) - from calibration
    GPT2* model,
    int* batched_tokens,            // Input: (num_batches, B, T) - batched calibration tokens
    int num_batches,                // Total batches
    int num_alpha_values,           // Number of alpha values (11)
    const float* alpha_values)      // Array [0.0, 0.1, ..., 1.0]
{
    int L = model->config.num_layers;
    int C = model->config.channels;
    int B = 32;  // Batch size for grid search
    int T = model->config.max_seq_len;
    
    GridSearchConfig config;
    config.L = L;
    config.C = C;
    config.B = B;
    config.T = T;
    config.num_alpha_values = num_alpha_values;
    config.alpha_values = alpha_values;
    
    printf("\n+-----------------------------+\n");
    printf("| AWQ AUTO-SCALING GRID SEARCH |\n");
    printf("+-----------------------------+\n");
    printf("Layers: %d, Channels: %d\n", L, C);
    printf("Batch size: %d, Num batches: %d\n", B, num_batches);
    printf("Alpha values: %d\n", num_alpha_values);
    printf("\n");
    
    // CRITICAL: Do a warm-up forward pass to ensure activations are allocated
    // with the correct B,T dimensions. This prevents reallocation during grid search,
    // which would invalidate activation pointers captured by get_matrix_pointers().
    printf("Warm-up forward pass to allocate activations (B=%d, T=%d)...\n", B, T);
    gpt2_forward(model, batched_tokens, B, T);
    printf("Activations allocated successfully.\n\n");
    
    // Allocate temporary buffers
    float* d_matrix_avgs;
    float* d_matrix_scales;
    cudaCheck(cudaMalloc(&d_matrix_avgs, L * 4 * C * sizeof(float)));  // Max size
    cudaCheck(cudaMalloc(&d_matrix_scales, L * 4 * C * sizeof(float)));
    
    // Process each matrix type
    MatrixType matrix_types[] = {MATRIX_QKV, MATRIX_ATTPROJ, MATRIX_FC1, MATRIX_FC2};
    const char* matrix_names[] = {"QKV", "AttProj", "FC1", "FC2"};
    size_t scale_offsets[] = {0, (size_t)(L*C), (size_t)(2*L*C), (size_t)(3*L*C)};
    size_t scale_sizes[] = {(size_t)(L*C), (size_t)(L*C), (size_t)(L*C), (size_t)(L*4*C)};
    
    for (int m = 0; m < 4; m++) {
        printf("\n========================================\n");
        printf("Processing %s matrix (type %d)\n", matrix_names[m], m);
        printf("========================================\n");
        
        // Extract channel averages for this matrix
        extract_channel_avgs_for_matrix(d_matrix_avgs, d_channel_avgs, 
                                       matrix_types[m], L, C);
        
        // Run grid search
        float* h_alphas = h_optimal_alphas + m * L;  // Pointer to this matrix's alphas
        grid_search_matrix(h_alphas, d_matrix_scales, d_matrix_avgs,
                          model, matrix_types[m], batched_tokens, 
                          num_batches, config);
        
        // Copy optimal scales to output
        cudaCheck(cudaMemcpy(d_optimal_scales + scale_offsets[m], 
                            d_matrix_scales, 
                            scale_sizes[m] * sizeof(float), 
                            cudaMemcpyDeviceToDevice));
    }
    
    // Cleanup
    cudaCheck(cudaFree(d_matrix_avgs));
    cudaCheck(cudaFree(d_matrix_scales));
    
    printf("\n========================================\n");
    printf("Grid search completed!\n");
    printf("========================================\n\n");
    
    // Apply optimal scales to model weights and biases
    apply_optimal_scales_to_model(d_optimal_scales, model);
}

// Apply optimal AWQ scales to model weights and biases in-place
void apply_optimal_scales_to_model(
    const float* d_optimal_scales,    // Input: (L*C*7) - optimal scales
    GPT2* model)
{
    int L = model->config.num_layers;
    int C = model->config.channels;
    
    printf("\n========================================\n");
    printf("Applying optimal scales to model...\n");
    printf("========================================\n");
    
    // Extract scale pointers for each matrix type
    const float* d_scales_ln1 = d_optimal_scales;                    // (L, C)
    const float* d_scales_atty = d_optimal_scales + L * C;           // (L, C)
    const float* d_scales_ln2 = d_optimal_scales + 2 * L * C;        // (L, C)
    const float* d_scales_fch_gelu = d_optimal_scales + 3 * L * C;   // (L, 4*C)
    
    // ========================================
    // MATRIX_QKV: scale_ln_fcs pattern
    // ln1.weight /= scales, ln1.bias /= scales
    // qkvw *= scales (input channels)
    // ========================================
    printf("  Scaling MATRIX_QKV (ln1 -> qkvw)...\n");
    
    // Scale ln1 parameters: divide by scales (treat as (L, 1, C))
    unscale_channels(model->params.ln1w, model->params.ln1w, d_scales_ln1, L, 1, C);
    unscale_channels(model->params.ln1b, model->params.ln1b, d_scales_ln1, L, 1, C);
    
    // Scale qkvw input channels: multiply by scales (L, 3*C, C)
    scale_channels(model->params.qkvw, model->params.qkvw, d_scales_ln1, L, 3*C, C);
    
    // ========================================
    // MATRIX_ATTPROJ: scale_fc_fc pattern (V portion only)
    // qkvw[2*C:3*C, :] /= scales (V weight output rows)
    // qkvb[2*C:3*C] /= scales (V bias)
    // attprojw *= scales (input channels)
    // ========================================
    printf("  Scaling MATRIX_ATTPROJ (V -> attprojw)...\n");
    
    // Scale V portion of qkvw: divide output rows by scales
    // qkvw shape: (L, 3*C, C), V weights are rows [2*C, 3*C)
    scale_fc_output_rows(model->params.qkvw, d_scales_atty, L, 3*C, C, 2*C, C);
    
    // Scale V portion of qkvb: divide by scales (treat as (L, 1, C) but offset to V bias)
    float* d_v_bias = model->params.qkvb + 2*C;  // Pointer to V bias section
    unscale_channels(d_v_bias, d_v_bias, d_scales_atty, L, 1, C);
    
    // Scale attprojw input channels: multiply by scales (L, C, C)
    scale_channels(model->params.attprojw, model->params.attprojw, d_scales_atty, L, C, C);
    
    // ========================================
    // MATRIX_FC1: scale_ln_fcs pattern
    // ln2.weight /= scales, ln2.bias /= scales
    // fcw *= scales (input channels)
    // ========================================
    printf("  Scaling MATRIX_FC1 (ln2 -> fcw)...\n");
    
    // Scale ln2 parameters: divide by scales (treat as (L, 1, C))
    unscale_channels(model->params.ln2w, model->params.ln2w, d_scales_ln2, L, 1, C);
    unscale_channels(model->params.ln2b, model->params.ln2b, d_scales_ln2, L, 1, C);
    
    // Scale fcw input channels: multiply by scales (L, 4*C, C)
    scale_channels(model->params.fcw, model->params.fcw, d_scales_ln2, L, 4*C, C);
    
    // ========================================
    // MATRIX_FC2: scale_fc_fc pattern
    // fcw /= scales (output rows, all 4*C rows)
    // fcb /= scales
    // fcprojw *= scales (input channels)
    // ========================================
    printf("  Scaling MATRIX_FC2 (fch_gelu -> fcprojw)...\n");
    
    // Scale fcw output rows: divide by scales (L, 4*C, C)
    scale_fc_output_rows(model->params.fcw, d_scales_fch_gelu, L, 4*C, C, 0, 4*C);
    
    // Scale fcb: divide by scales (treat as (L, 1, 4*C))
    unscale_channels(model->params.fcb, model->params.fcb, d_scales_fch_gelu, L, 1, 4*C);
    
    // Scale fcprojw input channels: multiply by scales (L, C, 4*C)
    scale_channels(model->params.fcprojw, model->params.fcprojw, d_scales_fch_gelu, L, C, 4*C);
    
    printf("  Model scaling completed!\n");
    printf("========================================\n\n");
}

#endif // __GRID_SEARCH_KERNEL_CUH__
