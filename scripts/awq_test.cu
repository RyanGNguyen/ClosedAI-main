//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------
// AWQ Model Testing and Comparison
// Compares standard GPT2 (FP32) vs AWQ quantized model performance
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

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../utils/utils.h"
#include "../utils/cuda_utils.cuh"
#include "../utils/tokenizer.h"
#include "../utils/logits.cuh"

#include "../gpt2.cuh"
#include "../gpt2_awq.cuh"

const char *output_filename = "awq_comparison_output.txt";
// const char *checkpoint_file_standard = "/work/hdd/bche/Project_GPT/gpt2_124M.bin";
// const char *checkpoint_file_awq = "/projects/bche/rnguyen2/gpt2_124M_awq.bin";
// const char *scales_checkpoint_awq = "/projects/bche/rnguyen2/awq_optimal_scales.bin";
// const char *tokenizer_file = "/work/hdd/bche/Project_GPT/gpt2_tokenizer.bin";
const char *checkpoint_file_standard = "/content/drive/MyDrive/gpt2_124M.bin";
const char *checkpoint_file_awq = "/content/fa25_ece408_ClosedAI-salient_awq/gpt2_124M_awq.bin";
const char *scales_checkpoint_awq = "/content/fa25_ece408_ClosedAI-salient_awq/awq_optimal_scales.bin";
const char *tokenizer_file = "/content/drive/MyDrive/gpt2_tokenizer.bin";

#define SEQUENCE_GENERATION_BATCH_SIZE 1  /* 1 for sequence generation */
#define TEMPERATURE 0.7  /* Should be > 0 */


void append_to_file(const char *filename, const char *string_to_append) {
    FILE *file = fopenCheck(filename, "a");
    fprintf(file, "%s", string_to_append);
    fclose(file);
}

void clear_output_file(const char *filename) {
    FILE *file = fopenCheck(filename, "w");
    fclose(file);
}

// Generic config accessor
template<typename ModelType>
struct ModelTraits;

template<>
struct ModelTraits<GPT2> {
    static int get_max_seq_len(const GPT2* m) { return m->config.max_seq_len; }
    static int get_vocab_size(const GPT2* m) { return m->config.vocab_size; }
    static int get_padded_vocab_size(const GPT2* m) { return m->config.padded_vocab_size; }
    static float* get_output(GPT2* m) { return m->acts.output; }
    static void forward(GPT2* m, int* inputs, int B, int T) { gpt2_forward(m, inputs, B, T); }
};

template<>
struct ModelTraits<GPT2_AWQ> {
    static int get_max_seq_len(const GPT2_AWQ* m) { return m->config.max_seq_len; }
    static int get_vocab_size(const GPT2_AWQ* m) { return m->config.vocab_size; }
    static int get_padded_vocab_size(const GPT2_AWQ* m) { return m->config.padded_vocab_size; }
    static float* get_output(GPT2_AWQ* m) { return m->acts.output; }
    static void forward(GPT2_AWQ* m, int* inputs, int B, int T) { gpt2_awq_forward(m, inputs, B, T); }
};

// Generic generation function
template<typename ModelType>
void generate_next_token(ModelType* model, int* input_tokens, int input_length, 
                        int* output_token, float** logits_out) {
    typedef ModelTraits<ModelType> Traits;
    
    assert(input_length <= Traits::get_max_seq_len(model));

    int padded_input_length = (input_length + 3) / 4 * 4;
    int * gpt_input = (int *)mallocCheck(padded_input_length * sizeof(int));
    memcpy(gpt_input, input_tokens, input_length * sizeof(int));
    if (padded_input_length - input_length > 0)
        memset(gpt_input + input_length, 0, (padded_input_length - input_length) * sizeof(int));
    
    Traits::forward(model, gpt_input, SEQUENCE_GENERATION_BATCH_SIZE, padded_input_length);

    int padded_vocab_size = Traits::get_padded_vocab_size(model);
    int vocab_size = Traits::get_vocab_size(model);
    float* logits_gpu = Traits::get_output(model) + (input_length - 1) * padded_vocab_size;
    float* logits_cpu = (float*)mallocCheck(vocab_size * sizeof(float));
    cudaCheck(cudaMemcpy(logits_cpu, logits_gpu, vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    *output_token = sample_next_token_temperature(logits_cpu, vocab_size, TEMPERATURE);
    
    if (logits_out != NULL) {
        *logits_out = logits_cpu;  // Caller must free
    } else {
        free(logits_cpu);
    }
    
    free(gpt_input);
}

template<typename ModelType>
char* generate_text(ModelType* model, Tokenizer* tokenizer, const char* input_sequence, 
                   int max_gen_length, int** all_tokens_out, int* total_length_out,
                   float** all_logits_out) {
    typedef ModelTraits<ModelType> Traits;
    
    int num_input_tokens = 0;
    int* input_tokens = tokenizer_encode(tokenizer, input_sequence, &num_input_tokens);
    
    printf("AWQ_TEST - Number of input tokens: %d\n", num_input_tokens);

    int max_seq_len = Traits::get_max_seq_len(model);
    int vocab_size = Traits::get_vocab_size(model);
    
    int* generated_sequence = (int*)mallocCheck((num_input_tokens + max_gen_length) * sizeof(int));
    memcpy(generated_sequence, input_tokens, num_input_tokens * sizeof(int));

    // Store all logits if requested (for perplexity calculation)
    float* all_logits = NULL;
    if (all_logits_out != NULL) {
        all_logits = (float*)mallocCheck(max_gen_length * vocab_size * sizeof(float));
    }

    int seq_len_itr = num_input_tokens;

    for (int i = 0; i < max_gen_length; ++i) {
        int next_token;
        float* logits = NULL;
        generate_next_token(model, generated_sequence, seq_len_itr, &next_token, 
                           (all_logits_out != NULL) ? &logits : NULL);

        if (all_logits != NULL && logits != NULL) {
            memcpy(all_logits + i * vocab_size, logits, vocab_size * sizeof(float));
            free(logits);
        }

        if (seq_len_itr < max_seq_len) {
            generated_sequence[seq_len_itr] = next_token;
            seq_len_itr += 1;
        }
    }

    // Build output text string
    size_t buffer_size = max_gen_length * 64;
    char* output_text = (char*)mallocCheck(buffer_size);
    output_text[0] = '\0';
    
    for (int i = num_input_tokens; i < num_input_tokens + max_gen_length; ++i) {
        const char* next_token_str = tokenizer_decode(tokenizer, generated_sequence[i]);
        strcat(output_text, next_token_str);
    }

    // Return tokens if requested
    if (all_tokens_out != NULL && total_length_out != NULL) {
        *all_tokens_out = generated_sequence;
        *total_length_out = seq_len_itr;
    } else {
        free(generated_sequence);
    }
    
    if (all_logits_out != NULL) {
        *all_logits_out = all_logits;
    }

    free(input_tokens);
    return output_text;
}

void log_comparison_header(const char* input_sequence) {
    append_to_file(output_filename, "═══════════════════════════════════════════════════════════════════\n");
    append_to_file(output_filename, "               AWQ MODEL COMPARISON TEST\n");
    append_to_file(output_filename, "═══════════════════════════════════════════════════════════════════\n\n");
    
    append_to_file(output_filename, "INPUT TEXT:\n");
    append_to_file(output_filename, input_sequence);
    append_to_file(output_filename, "\n\n");
}

void log_model_output(const char* model_name, const char* output_text, double perplexity, 
                      double time_ms, int num_tokens) {
    char buffer[4096];
    
    snprintf(buffer, sizeof(buffer), "───────────────────────────────────────────────────────────────────\n");
    append_to_file(output_filename, buffer);
    
    snprintf(buffer, sizeof(buffer), "%s MODEL OUTPUT:\n", model_name);
    append_to_file(output_filename, buffer);
    
    snprintf(buffer, sizeof(buffer), "───────────────────────────────────────────────────────────────────\n");
    append_to_file(output_filename, buffer);
    
    append_to_file(output_filename, output_text);
    append_to_file(output_filename, "\n\n");
    
    snprintf(buffer, sizeof(buffer), "Metrics:\n");
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Perplexity: %.6f\n", perplexity);
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Generation time: %.2f ms\n", time_ms);
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Tokens/sec: %.2f\n\n", (num_tokens * 1000.0) / time_ms);
    append_to_file(output_filename, buffer);
}

void log_parameter_comparison(GPT2* std_model, GPT2_AWQ* awq_model) {
    char buffer[1024];
    
    size_t std_size_bytes = std_model->num_parameters * sizeof(float);
    size_t awq_fp32_bytes = (awq_model->num_parameters - awq_model->num_parameters_quantized) * sizeof(float);
    size_t awq_int8_bytes = awq_model->num_parameters_quantized;
    size_t awq_total_bytes = awq_fp32_bytes + awq_int8_bytes;
    
    size_t std_size_mb = std_size_bytes >> 20;
    size_t awq_size_mb = awq_total_bytes >> 20;
    double compression_ratio = (double)std_size_bytes / awq_total_bytes;
    double size_reduction_pct = (1.0 - (double)awq_total_bytes / std_size_bytes) * 100.0;
    
    append_to_file(output_filename, "═══════════════════════════════════════════════════════════════════\n");
    append_to_file(output_filename, "              MODEL PARAMETER SIZE COMPARISON\n");
    append_to_file(output_filename, "═══════════════════════════════════════════════════════════════════\n\n");
    
    snprintf(buffer, sizeof(buffer), "Standard GPT2 (FP32):\n");
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Total parameters: %zu\n", std_model->num_parameters);
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Model size: %zu MiB\n\n", std_size_mb);
    append_to_file(output_filename, buffer);
    
    snprintf(buffer, sizeof(buffer), "AWQ Quantized Model:\n");
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Total parameters: %zu\n", awq_model->num_parameters);
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - FP32 parameters: %zu (%zu MiB)\n", 
             awq_model->num_parameters - awq_model->num_parameters_quantized, 
             awq_fp32_bytes >> 20);
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - INT8 parameters: %zu (%zu MiB)\n", 
             awq_model->num_parameters_quantized, awq_int8_bytes >> 20);
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Total size: %zu MiB\n\n", awq_size_mb);
    append_to_file(output_filename, buffer);
    
    snprintf(buffer, sizeof(buffer), "Compression Summary:\n");
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Compression ratio: %.2fx\n", compression_ratio);
    append_to_file(output_filename, buffer);
    snprintf(buffer, sizeof(buffer), "  - Size reduction: %.1f%%\n\n", size_reduction_pct);
    append_to_file(output_filename, buffer);
    
    append_to_file(output_filename, "═══════════════════════════════════════════════════════════════════\n\n");
    
    // Also print to console
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║              MODEL PARAMETER SIZE COMPARISON                      ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════╣\n");
    printf("║ Standard GPT2 (FP32)                                              ║\n");
    printf("║   Total parameters: %-45zu ║\n", std_model->num_parameters);
    printf("║   Model size: %-50zu MiB ║\n", std_size_mb);
    printf("║                                                                   ║\n");
    printf("║ AWQ Quantized Model                                               ║\n");
    printf("║   Total parameters: %-45zu ║\n", awq_model->num_parameters);
    printf("║   FP32 parameters: %-33zu (%zu MiB) ║\n", 
           awq_model->num_parameters - awq_model->num_parameters_quantized,
           awq_fp32_bytes >> 20);
    printf("║   INT8 parameters: %-33zu (%zu MiB) ║\n",
           awq_model->num_parameters_quantized, awq_int8_bytes >> 20);
    printf("║   Total size: %-46zu MiB ║\n", awq_size_mb);
    printf("║                                                                   ║\n");
    printf("║ Compression Summary                                               ║\n");
    printf("║   Compression ratio: %-44.2fx ║\n", compression_ratio);
    printf("║   Size reduction: %-46.1f%% ║\n", size_reduction_pct);
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("AWQ_TEST - Usage: %s [--compare|-c] \"<input sequence>\"\n", argv[0]);
        printf("  Default: AWQ-only profiling mode (ideal for nsys/ncu profiling)\n");
        printf("  --compare, -c: Run comparison between standard GPT2 and AWQ models\n");
        return EXIT_FAILURE;
    }

    // Parse command-line arguments
    bool compare_mode = false;
    const char* input_sequence = NULL;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--compare") == 0 || strcmp(argv[i], "-c") == 0) {
            compare_mode = true;
        } else {
            input_sequence = argv[i];
        }
    }
    
    if (input_sequence == NULL) {
        printf("AWQ_TEST - Error: No input sequence provided\n");
        printf("AWQ_TEST - Usage: %s [--compare|-c] \"<input sequence>\"\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(time(NULL));

    // Setup cuBLAS
    cublasCheck(cublasCreate(&cublas_handle));
    int enable_tf32 = 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    printf("\n");
    if (compare_mode) {
        printf("╔═══════════════════════════════════════════════════════════════════╗\n");
        printf("║           AWQ MODEL COMPARISON TEST                               ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    } else {
        printf("╔═══════════════════════════════════════════════════════════════════╗\n");
        printf("║           AWQ MODEL PROFILING                                     ║\n");
        printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
    }

    // Initialize tokenizer
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, tokenizer_file);

    int max_gen_length = 50;

    // Conditional model loading and generation
    GPT2 std_model;
    bool std_model_loaded = false;
    
    if (compare_mode) {
        // Load standard GPT2 model for comparison
        printf("Loading standard GPT2 model...\n");
        gpt2_build_from_checkpoint(&std_model, checkpoint_file_standard);
        std_model_loaded = true;
        
        printf("+-----------------------+----------------------------------------------------+\n");
        printf("| Standard GPT2 Model   |                                                    |\n");
        printf("+-----------------------+----------------------------------------------------+\n");
        printf("| num_parameters        | %-50zu |\n", std_model.num_parameters);
        printf("| Model size (MiB)      | %-50zu |\n", (std_model.num_parameters * 4) >> 20);
        printf("+-----------------------+----------------------------------------------------+\n\n");
    }

    // Load AWQ model (always)
    printf("Loading AWQ quantized model...\n");
    GPT2_AWQ awq_model;
    gpt2_awq_build_from_checkpoint(&awq_model, checkpoint_file_awq);
    load_awq_scales(&awq_model, scales_checkpoint_awq);
    printf("\n");

    // Clear output file and setup header
    clear_output_file(output_filename);
    
    if (compare_mode) {
        log_comparison_header(input_sequence);
        printf("Generating text with both models...\n\n");
    } else {
        append_to_file(output_filename, "═══════════════════════════════════════════════════════════════════\n");
        append_to_file(output_filename, "               AWQ MODEL PROFILING\n");
        append_to_file(output_filename, "═══════════════════════════════════════════════════════════════════\n\n");
        append_to_file(output_filename, "INPUT TEXT:\n");
        append_to_file(output_filename, input_sequence);
        append_to_file(output_filename, "\n\n");
        printf("Generating text with AWQ model...\n\n");
    }

    struct timespec start, end;
    
    // Generate with standard model if in comparison mode
    char* std_output = NULL;
    int* std_tokens = NULL;
    int std_total_length = 0;
    float* std_logits = NULL;
    double std_time_ms = 0.0;
    double std_perplexity = 0.0;
    
    if (compare_mode) {
        printf("► Standard GPT2 model generating...\n");
        clock_gettime(CLOCK_MONOTONIC, &start);
        std_output = generate_text(&std_model, &tokenizer, input_sequence, max_gen_length,
                                   &std_tokens, &std_total_length, &std_logits);
        clock_gettime(CLOCK_MONOTONIC, &end);
        std_time_ms = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1000.0;

        // Calculate perplexity for standard model
        std_perplexity = calculate_perplexity(std_logits, 
                                              std_tokens + (std_total_length - max_gen_length),
                                              max_gen_length - 1,
                                              std_model.config.vocab_size);

        printf("  Generated in %.2f ms\n", std_time_ms);
        print_perplexity("Standard GPT2", std_perplexity, max_gen_length - 1);
        printf("\n");

        log_model_output("STANDARD GPT2", std_output, std_perplexity, std_time_ms, max_gen_length);
    }

    // Generate with AWQ model (always)
    printf("► AWQ quantized model generating...\n");
    int* awq_tokens = NULL;
    int awq_total_length = 0;
    float* awq_logits = NULL;
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    char* awq_output = generate_text(&awq_model, &tokenizer, input_sequence, max_gen_length,
                                     &awq_tokens, &awq_total_length, &awq_logits);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double awq_time_ms = ((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1000.0;

    // Calculate perplexity for AWQ model
    double awq_perplexity = calculate_perplexity(awq_logits, 
                                                  awq_tokens + (awq_total_length - max_gen_length),
                                                  max_gen_length - 1,
                                                  awq_model.config.vocab_size);

    printf("  Generated in %.2f ms\n", awq_time_ms);
    print_perplexity("AWQ Model", awq_perplexity, max_gen_length - 1);
    printf("\n");

    log_model_output("AWQ QUANTIZED", awq_output, awq_perplexity, awq_time_ms, max_gen_length);

    // Log parameter comparison if in comparison mode
    if (compare_mode) {
        log_parameter_comparison(&std_model, &awq_model);
    }
    
    printf("Results saved to: %s\n\n", output_filename);

    // Cleanup
    if (compare_mode) {
        free(std_output);
        free(std_tokens);
        free(std_logits);
    }
    free(awq_output);
    free(awq_tokens);
    free(awq_logits);
    
    tokenizer_free(&tokenizer);
    if (std_model_loaded) {
        gpt2_free(&std_model);
    }
    gpt2_awq_free(&awq_model);
    cublasDestroy(cublas_handle);

    return EXIT_SUCCESS;
}
