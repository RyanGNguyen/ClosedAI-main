//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#include <cstring>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <float.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <utility>

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../utils/utils.h"
#include "../utils/tokenizer.h"
#include "../utils/logits.cuh"
#include "../utils/cuda_utils.cuh"

#include "../gpt2.cuh"
#include "../kernels_awq/saliency.cuh"

const char *calibration_output_filename = "calibration_output.txt";
const char *channel_averages_filename = "awq_channel_avgs.bin";
const char *salient_indices_filename = "salient_channel_indices.bin";
const char *calibration_tokens_filename = "calibration_tokens.bin";
const char *checkpoint_file = "/content/drive/MyDrive/gpt2_124M.bin";
#define CALIBRATION_BATCH_SIZE 32  /* Batch size for calibration */
#define TEMPERATURE 0.7  /* Should be > 0 */

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void append_to_file(const char *filename, const char *string_to_append) {
    FILE *file = fopenCheck(filename, "a");
    if (file == NULL) {
        perror("next_token_gen - Error opening file");
        exit(EXIT_FAILURE);
    }
    fprintf(file, "%s", string_to_append);
    fclose(file);
}

void log_generation_output(const char* input_sequence, const char* output_text) {
    append_to_file(calibration_output_filename, "next_token_gen - INPUT TEXT:\n");
    append_to_file(calibration_output_filename, input_sequence);
    append_to_file(calibration_output_filename, "\n\nGENERATED TEXT:\n");
    append_to_file(calibration_output_filename, output_text);
    append_to_file(calibration_output_filename, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void save_channel_averages(const float* h_averages, int L, int C, size_t total_tokens, const char* output_file) {
    // Calculate total channels: L*C + L*C + L*C + L*4*C = L*C*7
    size_t total_channels = L * C * 7;
    
    // Write to binary file
    FILE* fp = fopenCheck(output_file, "wb");
    
    // Write header
    fwrite(&L, sizeof(int), 1, fp);
    fwrite(&C, sizeof(int), 1, fp);
    fwrite(&total_tokens, sizeof(size_t), 1, fp);
    
    // Write average magnitudes
    fwrite(h_averages, sizeof(float), total_channels, fp);
    
    fcloseCheck(fp);
    
    printf("next_token_gen - Saved %zu channel averages to %s\n", total_channels, output_file);
}

void save_salient_indices(const float* h_averages, int L, int C, const char* output_file) {
    // Calculate k values (top 1%)
    int k_c = (int)ceil(0.01f * C);      // top 1% of C
    int k_4c = (int)ceil(0.01f * 4 * C); // top 1% of 4*C
    
    printf("next_token_gen - Computing top 1%% salient channel indices (k_c=%d, k_4c=%d)...\n", 
           k_c, k_4c);
    
    // Layout: [L*C ln1] [L*C atty] [L*C ln2] [L*4*C fch_gelu]
    size_t offset_ln1 = 0;
    size_t offset_atty = L * C;
    size_t offset_ln2 = 2 * L * C;
    size_t offset_fch_gelu = 3 * L * C;
    
    // Allocate host memory for indices
    uint32_t* h_salient_ln1 = (uint32_t*)mallocCheck(L * k_c * sizeof(uint32_t));
    uint32_t* h_salient_atty = (uint32_t*)mallocCheck(L * k_c * sizeof(uint32_t));
    uint32_t* h_salient_ln2 = (uint32_t*)mallocCheck(L * k_c * sizeof(uint32_t));
    uint32_t* h_salient_fch_gelu = (uint32_t*)mallocCheck(L * k_4c * sizeof(uint32_t));
    
    // CPU-based top-k selection for each layer and tensor type
    for (int l = 0; l < L; l++) {
        // Helper lambda for top-k selection on a segment
        auto extract_top_k = [&](uint32_t* out_indices, size_t segment_offset, int segment_size, int k) {
            std::vector<std::pair<uint32_t, float>> indexed_mags(segment_size);
            for (int i = 0; i < segment_size; i++) {
                indexed_mags[i] = std::make_pair(i, h_averages[segment_offset + i]);
            }
            
            // Partial sort to get top k (descending by magnitude)
            std::partial_sort(indexed_mags.begin(), 
                              indexed_mags.begin() + k,
                              indexed_mags.end(),
                              [](const std::pair<uint32_t, float>& a, 
                                 const std::pair<uint32_t, float>& b) {
                                  return a.second > b.second;  // descending
                              });
            
            // Extract top k indices
            for (int i = 0; i < k; i++) {
                out_indices[i] = indexed_mags[i].first;
            }
        };
        extract_top_k(h_salient_ln1 + l * k_c, offset_ln1 + l * C, C, k_c);
        extract_top_k(h_salient_atty + l * k_c, offset_atty + l * C, C, k_c);
        extract_top_k(h_salient_ln2 + l * k_c, offset_ln2 + l * C, C, k_c);
        extract_top_k(h_salient_fch_gelu + l * k_4c, offset_fch_gelu + l * 4 * C, 4 * C, k_4c);
    }
    
    // Write to binary file with contiguous layout
    FILE* fp = fopenCheck(output_file, "wb");
    
    // Write header
    fwrite(&L, sizeof(int), 1, fp);
    fwrite(&C, sizeof(int), 1, fp);
    fwrite(&k_c, sizeof(int), 1, fp);
    fwrite(&k_4c, sizeof(int), 1, fp);
    
    // Write indices in contiguous arrays
    fwrite(h_salient_ln1, sizeof(uint32_t), L * k_c, fp);       // L * k_c
    fwrite(h_salient_atty, sizeof(uint32_t), L * k_c, fp);      // L * k_c
    fwrite(h_salient_ln2, sizeof(uint32_t), L * k_c, fp);       // L * k_c
    fwrite(h_salient_fch_gelu, sizeof(uint32_t), L * k_4c, fp); // L * k_4c
    
    fcloseCheck(fp);
    
    size_t total_indices = L * (3 * k_c + k_4c);
    printf("next_token_gen - Saved %zu salient indices to %s\n", total_indices, output_file);
    
    // Cleanup
    free(h_salient_ln1);
    free(h_salient_atty);
    free(h_salient_ln2);
    free(h_salient_fch_gelu);
}

void gpt2_generate_next_token(GPT2* model, int* batched_sequences, int* seq_lens, 
                                       int B, int T, int V, int Vp) {
    // Single batched forward pass
    gpt2_forward(model, batched_sequences, B, T);
    
    // Sample next token for each sequence
    float* logits_cpu = (float*)mallocCheck(V * sizeof(float));
    int next_token;

    for (int b = 0; b < B; b++) {
        // Get logits at last sequence position for batch b
        // acts.output *should* be [B, T, Vp]. See gpt2.cuh for detail
        int logit_offset = (b * T + (seq_lens[b] - 1)) * Vp;
        float* logits_gpu = model->acts.output + logit_offset;
        
        // Copy logits to CPU (only vocab_size, not padded)
        cudaCheck(cudaMemcpy(logits_cpu, logits_gpu, V * sizeof(float), cudaMemcpyDeviceToHost));
        
        // Sample next token
        next_token = sample_next_token_temperature(logits_cpu, V, TEMPERATURE);
        
        // Add to output /
        if (seq_lens[b] < T) {
            batched_sequences[b * T + seq_lens[b]] = next_token;
            seq_lens[b]++;
        }
    }
    
    free(logits_cpu);
}

void gpt2_generate_text(GPT2* model, char** input_sequences, int B, int max_gen_length, 
                        size_t* sum_initial_tokens, FILE* tokens_file) {
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "/content/drive/MyDrive/gpt2_tokenizer.bin");
    
    // Fixed model parameters
    int T = model->config.max_seq_len; 
    int V = model->config.vocab_size;
    int Vp = model->config.padded_vocab_size;
    
    int** input_tokens = (int**)mallocCheck(B * sizeof(int*)); // All input tokens
    int* num_tokens = (int*)mallocCheck(B * sizeof(int)); // Need to store original count for decoding
    int* h_batched_seq = (int*)calloc(B * T, sizeof(int)); // Match GPU 1D indexing + initialized 0 padding
    
    // Process inputs
    for (int b = 0; b < B; b++) {
        // Tokenize inputs
        input_tokens[b] = tokenizer_encode(&tokenizer, input_sequences[b], &num_tokens[b]);

        // Token sizing. No need for padding due to fixed size at T.
        if (num_tokens[b] + max_gen_length > T) { // Truncate (leave room for generation)
            printf("next_token_gen - Warning: Batch %d (padded) input (%d tokens) + output (%d tokens) exceeds max_seq_len (%d). Truncating input.\n", 
                   b + 1, num_tokens[b], max_gen_length, T);
            num_tokens[b] = T - max_gen_length;
        }
        
        // Track initial token count (before generation)
        *sum_initial_tokens += num_tokens[b];
        
        // Copy accepted tokens to host array
        memcpy(h_batched_seq + b * T, input_tokens[b], num_tokens[b] * sizeof(int));
    }
    
    // Generation loop - generate max_gen_length tokens for all batched sequences
    for (int gen_step = 0; gen_step < max_gen_length; gen_step++) {
        gpt2_generate_next_token(model, h_batched_seq, num_tokens, B, T, V, Vp);
    }

    // Decode and save outputs for debugging, and write tokens to file
    for (int b = 0; b < B; b++) {
        // Write this sequence to tokens file (T ints, padding included)
        fwrite(h_batched_seq + b * T, sizeof(int), T, tokens_file);
        
        // Allocate buffer for accumulated output (generous size)
        size_t buffer_size = max_gen_length * 64;  // tokens are typically short
        char* output_string = (char*)mallocCheck(buffer_size);
        size_t current_len = 0;
        
        for (int i = num_tokens[b] - max_gen_length; i < num_tokens[b]; ++i) {
            const char* output_token = tokenizer_decode(&tokenizer, h_batched_seq[b * T + i]);
            size_t token_len = strlen(output_token);
            
            // Copy token if there's room
            if (current_len + token_len < buffer_size) {
                memcpy(output_string + current_len, output_token, token_len);
                current_len += token_len;
            }
        }
        output_string[current_len] = '\0';  // Null-terminate
        
        // Log once per batch with complete output
        log_generation_output(input_sequences[b], output_string);
        free(output_string);
    }
    
    // Cleanup
    for (int b = 0; b < B; b++) {
        free(input_tokens[b]);
    }
    free(input_tokens);
    free(num_tokens);
    free(h_batched_seq);
    tokenizer_free(&tokenizer);
}

void init_cublas() {
    cublasCheck(cublasCreate(&cublas_handle));
    int enable_tf32 = 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
}

void print_model_info(const GPT2* model) {
    printf("+-----------------------+\n");
    printf("| GPT2 MODEL PARAMETERS |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| max_sequence_length T | %-50d |\n", model->config.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model->config.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model->config.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model->config.num_layers);
    printf("| num_heads NH          | %-50d |\n", model->config.num_heads);
    printf("| channels C            | %-50d |\n", model->config.channels);
    printf("| num_parameters        | %-50zu |\n", model->num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");
}

void process_batch(GPT2* model, char** batch_sequences, int batch_count, 
                   int max_gen_length, int* sample_count, double* total_time_s,
                   size_t* sum_initial_tokens, FILE* tokens_file) {
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    gpt2_generate_text(model, batch_sequences, batch_count, max_gen_length, 
                      sum_initial_tokens, tokens_file);
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = 
        (end.tv_sec - start.tv_sec) +
        (end.tv_nsec - start.tv_nsec) / 1e9;
    
    *total_time_s += time_elapsed_s;
    *sample_count += batch_count;
    
    printf(">>> Processed %d samples (avg %.2f s/sample)\n",
        *sample_count, (*total_time_s / *sample_count));
    
    // Free batch memory
    for (int i = 0; i < batch_count; i++) {
        free(batch_sequences[i]);
    }
}

size_t run_calibration(GPT2* model, const char* samples_filepath, int max_gen_length) {
    FILE* samples_file = fopenCheck(samples_filepath, "r");
    FILE* tokens_file = fopenCheck(calibration_tokens_filename, "wb");
    
    char record[8192];
    int record_pos = 0;
    
    int sample_count = 0;
    double total_time_s = 0.0;
    size_t sum_initial_tokens = 0;

    printf("\n+---------------------------+\n");
    printf("| CALIBRATION RUN STARTED   |\n");
    printf("+---------------------------+\n");
    printf("Processing calibration samples from: %s\n", samples_filepath);
    printf("Batch size: %d\n\n", CALIBRATION_BATCH_SIZE);

    // Allocate batch storage
    char** batch_sequences = (char**)mallocCheck(CALIBRATION_BATCH_SIZE * sizeof(char*));
    int batch_count = 0;
    
    int c;
    
    while ((c = fgetc(samples_file)) != EOF) {
        
        if (c == '\x1e') {  // RECORD SEPARATOR FOUND
            if (record_pos > 0) {
                record[record_pos] = '\0';  // null-terminate the sample
                
                // Allocate and copy sample to batch
                batch_sequences[batch_count] = (char*)mallocCheck((record_pos + 1) * sizeof(char));
                strcpy(batch_sequences[batch_count], record);
                batch_count++;
                
                // Process batch when full
                if (batch_count == CALIBRATION_BATCH_SIZE) {
                    process_batch(model, batch_sequences, batch_count, max_gen_length, 
                                &sample_count, &total_time_s, &sum_initial_tokens, tokens_file);
                    batch_count = 0;
                }
            }
            
            // Reset for next record
            record_pos = 0;
        }
        else {
            // Regular character (including ALL newlines - they're all part of content now)
            if (record_pos < (int)sizeof(record) - 1) {
                record[record_pos++] = c;
            }
            else {
                // Buffer overflow warning (only warn once)
                static int overflow_warned = 0;
                if (!overflow_warned) {
                    fprintf(stderr, "Warning: Sample %d exceeds buffer size, truncating\n", 
                            sample_count + batch_count + 1);
                    overflow_warned = 1;
                }
            }
        }
    }
    
    // Process final record if file doesn't end with \x1e
    if (record_pos > 0) {
        record[record_pos] = '\0';
        
        // Allocate and copy sample to batch
        batch_sequences[batch_count] = (char*)mallocCheck((record_pos + 1) * sizeof(char));
        strcpy(batch_sequences[batch_count], record);
        batch_count++;
    }
    
    // Process final partial batch if any samples remain
    if (batch_count > 0) {
        process_batch(model, batch_sequences, batch_count, max_gen_length, 
                    &sample_count, &total_time_s, &sum_initial_tokens, tokens_file);
    }
    
    // Free batch array
    free(batch_sequences);
    fcloseCheck(samples_file);
    fcloseCheck(tokens_file);
    
    printf("next_token_gen - Saved calibration tokens to %s\n", calibration_tokens_filename);

    printf("\n+---------------------------+\n");
    printf("| CALIBRATION COMPLETE      |\n");
    printf("+---------------------------+------------------------------------------------------+\n");
    printf("| Total Samples    | %-52d |\n", sample_count);
    printf("| Total Time (s)   | %-52.2f |\n", total_time_s);
    printf("| Avg Time (ms)    | %-52.2f |\n", (total_time_s / sample_count) * 1000);
    printf("+---------------------------+------------------------------------------------------+\n\n");
    
    // Compute total token-passes using formula: G * sum_initial_tokens + sample_count * G * (G-1) / 2
    size_t G = (size_t)max_gen_length;
    size_t total_tokens = G * sum_initial_tokens + (size_t)sample_count * G * (G - 1) / 2;
    
    printf("next_token_gen - Total token-passes computed: %zu (G=%zu, initial_sum=%zu, samples=%d)\n",
           total_tokens, G, sum_initial_tokens, sample_count);
    
    return total_tokens;
}

void finalize_and_save_calibration(GPT2* model, size_t total_tokens) {
    gpt2_disable_calibration(model);
    
    // Finalize averages and copy to host once
    int L = model->config.num_layers;
    int C = model->config.channels;
    size_t total_channels = L * C * 7;  // L*C + L*C + L*C + L*4*C
    
    printf("next_token_gen - Finalizing channel averages (total tokens: %zu)...\n", total_tokens);
    finalize_averages(model->saliency_memory, total_channels, total_tokens);
    
    // Copy to host
    float* h_averages = (float*)mallocCheck(total_channels * sizeof(float));
    cudaCheck(cudaMemcpy(h_averages, model->saliency_memory, 
                         total_channels * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Save channel averages and salient indices
    save_channel_averages(h_averages, L, C, total_tokens, channel_averages_filename);
    save_salient_indices(h_averages, L, C, salient_indices_filename);
    
    // Cleanup
    free(h_averages);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("next_token_gen - Usage: %s <filepath_to_samples.txt>\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(time(NULL));
    init_cublas();

    GPT2 model;
    gpt2_build_from_checkpoint(&model, checkpoint_file);
    gpt2_enable_calibration(&model);
    print_model_info(&model);

    int max_gen_length = 10;
    
    // Run calibration (tokens written to file during processing)
    size_t total_tokens = run_calibration(&model, argv[1], max_gen_length);
    
    finalize_and_save_calibration(&model, total_tokens);

    gpt2_free(&model);

    return EXIT_SUCCESS;
}
