//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils/utils.h"
#include "utils/tokenizer.h"
#include "utils/cuda_utils.cuh"

// GPT-2 types
#include "utils/gpt2_types.cuh"

// kernels 
#include "kernels/attention.cuh"
#include "kernels/encoder.cuh"
#include "kernels/gelu.cuh"
#include "kernels/layernorm.cuh"
#include "kernels/matmul.cuh"
#include "kernels/residual.cuh"

// AWQ kernels
#include "kernels_awq/saliency.cuh"

void gpt2_build_from_checkpoint(GPT2 *model, const char* checkpoint_path) {

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { fprintf(stderr, "GPT2 - Bad magic model file\n"); exit(EXIT_FAILURE); }
    if (model_header[1] != 3) {
        // was bumped from 1 -> 3 to incorporate the padded vocab size
        fprintf(stderr, "GPT2 - Bad version in model file\n");
        exit(EXIT_FAILURE);
    }

    // read in hyperparameters
    model->config.max_seq_len = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.channels = model_header[6];
    model->config.padded_vocab_size = model_header[7];

    // allocate space for all the parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);

    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters += model->param_sizes[i];
    }
    model->num_parameters = num_parameters;

    // create memory for model parameters on the device
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_sizes, 1);

    // read in all the parameters from file and copy them to device
    float* params_memory_cpu = (float*)mallocCheck(num_parameters * sizeof(float));
    freadCheck(params_memory_cpu, sizeof(float), num_parameters, model_file);
    cudaCheck(cudaMemcpy(model->params_memory, params_memory_cpu, num_parameters * sizeof(float), cudaMemcpyHostToDevice));
    free(params_memory_cpu);
    fcloseCheck(model_file);

    // other inits
    model->acts_memory = NULL;
    model->grads_memory = NULL;
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->grads_acts_memory = NULL;
    model->inputs = NULL;
    model->targets = NULL;
    model->cpu_losses = NULL;
    model->batch_size = 0;
    model->seq_len = 0;
    model->mean_loss = -1.0f; // -1.0f will designate no loss
    
    // Initialize calibration fields
    model->calibration_enabled = 0;
    model->saliency_memory = NULL;
}

// Enable AWQ calibration mode
// Allocates memory for accumulating channel magnitudes
void gpt2_enable_calibration(GPT2* model) {
    int L = model->config.num_layers;
    int C = model->config.channels;
    
    // Allocate saliency memory: L*C + L*C + L*C + L*4*C = L*C*7
    size_t total_channels = L * C * 7;
    cudaCheck(cudaMalloc(&model->saliency_memory, total_channels * sizeof(float)));
    cudaCheck(cudaMemset(model->saliency_memory, 0, total_channels * sizeof(float)));
    
    model->calibration_enabled = 1;
    printf("GPT2 - Calibration enabled (%zu channels)\n", total_channels);
}

// Disable AWQ calibration mode
// Keeps saliency_memory allocated for retrieval
void gpt2_disable_calibration(GPT2* model) {
    model->calibration_enabled = 0;
    printf("GPT2 - Calibration disabled\n");
}

void gpt2_forward(GPT2 *model, int* inputs, int B, int T) {

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("GPT2 - Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;
    int C = model->config.channels;

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= inputs[i] && inputs[i] < V);
    }

    // allocate space for all the activations if needed (done here, lazily)
    if(model->acts_memory == NULL || B != model->batch_size || T != model->seq_len) {
        // record the current B,T as well
        model->batch_size = B;
        model->seq_len = T;
        // and now allocate the space
        fill_in_activation_sizes(model->act_sizes, B, T, model->config);
        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
            num_activations += model->act_sizes[i];
        }
        model->num_activations = num_activations;
        if (model->acts_memory != NULL) {
            printf("GPT2 - Reallocating activations for B=%d T=%d\n", B, T);
            cudaFree(model->acts_memory);
            cudaFree(model->inputs);
        }
        model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
        printf("GPT2 - Allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
        cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
        cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));
    }

    // copy inputs to the model
    cudaCheck(cudaMemcpy(model->inputs, inputs, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]

    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
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
        // these are only needed as scratchpads for the forward pass
        float* scratch = acts.output;

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    matmul_forward(acts.output, acts.lnf, params.wte, NULL, B, T, C, Vp);

    // AWQ calibration: accumulate channel magnitudes if enabled
    if (model->calibration_enabled) {
        int N = B * T;
        // Layout: [ln1 (L*C), atty (L*C), ln2 (L*C), fch_gelu (L*4*C)]
        accumulate_salient_channels(model->saliency_memory,             acts.ln1, L, N, C);
        accumulate_salient_channels(model->saliency_memory + L*C,       acts.atty, L, N, C);
        accumulate_salient_channels(model->saliency_memory + 2*L*C,     acts.ln2, L, N, C);
        accumulate_salient_channels(model->saliency_memory + 3*L*C,     acts.fch_gelu, L, N, 4*C);
    }

    model->mean_loss = -1.0f;
}

void gpt2_free(GPT2 *model) {
    cudaCheck(cudaFree(model->params_memory));
    cudaCheck(cudaFree(model->grads_memory));
    cudaCheck(cudaFree(model->m_memory));
    cudaCheck(cudaFree(model->v_memory));
    cudaCheck(cudaFree(model->acts_memory));
    cudaCheck(cudaFree(model->grads_acts_memory));
    cudaCheck(cudaFree(model->inputs));
    cudaCheck(cudaFree(model->targets));
    cudaFreeHost(model->cpu_losses);
    // Free AWQ calibration memory if allocated
    if (model->saliency_memory != NULL) {
        cudaCheck(cudaFree(model->saliency_memory));
    }
}
