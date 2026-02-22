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

// GPU / CUDA related
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utils/utils.h"
#include "utils/tokenizer.h"
#include "utils/cuda_utils.cuh"

#include "gpt2.cuh"

const char *generation_output_filename = "generation_outputs.txt";
const char *checkpoint_file = "/work/hdd/bche/Project_GPT/gpt2_124M.bin";
#define SEQUENCE_GENERATION_BATCH_SIZE 1  /* 1 for sequence generation */
#define TEMPERATURE 0.7  /* Should be > 0 */

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ HELPER FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void get_substring_up_to_space(char *input, char *output) {
    char *space_pos = strchr(input + 1, ' ');

    if (space_pos != NULL) {
        size_t length = space_pos - input;
        strncpy(output, input, length);
        // null-terminate output
        output[length] = '\0';
    } else {
        // if no space, copy the entire string
        strcpy(output, input);
    }
}

int* tokenizer_encode(Tokenizer *tokenizer, const char *input_str, int *num_tokens) {
    if (tokenizer->init_ok == 0) {
        *num_tokens = 0;
        return NULL;
    }

    // arbitrary initial number of tokens
    int max_tokens = 128;
    int *tokens = (int*)mallocCheck(max_tokens * sizeof(int));
    int token_count = 0;

    char temp_str[strlen(input_str) + 1];
    strcpy(temp_str, input_str);

    // pointer to keep track of the remaining string to process
    char *remaining_input_str = temp_str;
    while (*remaining_input_str != '\0') {
        char to_search[100];
        get_substring_up_to_space(remaining_input_str, to_search);
        size_t token_len = strlen(to_search);

        // try to find longest matching token from the token_table (not perfect, but probably good enough heuristic)
        int best_token = -1;
        int best_token_length = -1;
        for (int i = 0; i < tokenizer->vocab_size; ++i) {
            const char *token = tokenizer->token_table[i];
            int tokenlen = strlen(token);

            if (strncmp(to_search, token, tokenlen) == 0) {
                if (tokenlen > best_token_length) {
                    best_token_length = tokenlen;
                    best_token = i;
                }
            }
        }
        assert(best_token != 0);
        tokens[token_count++] = best_token;
        // move the pointer ahead by the length of the token
        remaining_input_str += best_token_length;

        // if reached current tokens array capacity, resize
        if (token_count >= max_tokens) {
            // double the capacity
            max_tokens *= 2;
            tokens = (int*)realloc(tokens, max_tokens * sizeof(int));
            if (!tokens) {
                fprintf(stderr, "next_token_gen - Memory allocation failed during token array resizing.\n");
                exit(EXIT_FAILURE);
            }
        }
    }

    *num_tokens = token_count;
    return tokens;
}

int return_random_token(int vocab_size) {
    int lower_bound = 0;
    int upper_bound = vocab_size - 1;
    return rand() % (upper_bound - lower_bound + 1) + lower_bound;
}

int sample_next_token_temperature(const float* logits, int vocab_size, float temperature) {
    // find max logit, to be used for numerical stability
    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    float norm = 0.0f;
    float* probabilities = (float*)mallocCheck(vocab_size * sizeof(float));

    // compute probabilities using softmax with numerical stability and temperature
    for (int i = 0; i < vocab_size; i++) {
        probabilities[i] = expf((logits[i] - max_logit) / temperature);
        norm += probabilities[i];
    }

    // normalize
    for (int i = 0; i < vocab_size; i++) {
        probabilities[i] /= norm;
    }

    // sample from distribution
    float random_value = (float)rand() / RAND_MAX;
    float cumulative = 0.0f;

    for (int i = 0; i < vocab_size; ++i) {
        cumulative += probabilities[i];
        if (random_value < cumulative) {
            free(probabilities);
            return i;
        }
    }

    free(probabilities);

    // default fallback
    return vocab_size - 1; // return last token in vocab
    // return return_random_token(vocab_size); // return a random token in vocab
}

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
    append_to_file(generation_output_filename, "next_token_gen - INPUT TEXT:\n");
    append_to_file(generation_output_filename, input_sequence);
    append_to_file(generation_output_filename, "\n\nGENERATED TEXT:\n");
    append_to_file(generation_output_filename, output_text);
    append_to_file(generation_output_filename, "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

void gpt2_generate_next_token(GPT2* model, int* input_tokens, int input_length, int* output_token) {
    assert(input_length <= model->config.max_seq_len);

	int padded_input_length = (input_length + 3) / 4 * 4;
	int * gpt_input = (int *)mallocCheck(padded_input_length * sizeof(int));
	memcpy(gpt_input, input_tokens, input_length * sizeof(int));
	if (padded_input_length - input_length > 0)
		memset(gpt_input + input_length, 0, (padded_input_length - input_length) * sizeof(int));
    gpt2_forward(model, gpt_input, SEQUENCE_GENERATION_BATCH_SIZE, padded_input_length);

    float* logits_gpu = (model->acts.output) + (input_length - 1) * model->config.padded_vocab_size;
    float* logits_cpu = (float*)mallocCheck(model->config.vocab_size * sizeof(float));
    // move logits back to CPU and sample (note we only move the first vocab_size logits, ignoring the padding)
    cudaCheck(cudaMemcpy(logits_cpu, logits_gpu, model->config.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));

    *output_token = sample_next_token_temperature(logits_cpu, model->config.vocab_size, TEMPERATURE);    

    free(logits_cpu);
}

void gpt2_generate_text(GPT2* model, const char* input_sequence, int max_gen_length, char* output_text) {
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "/work/hdd/bche/Project_GPT/gpt2_tokenizer.bin");

    int num_input_tokens = 0;
    int* input_tokens = tokenizer_encode(&tokenizer, input_sequence, &num_input_tokens);
    printf("next_token_gen - number of input tokens: %d\n", num_input_tokens);
    for (int i = 0; i < num_input_tokens; i++) {
        printf("%d ", input_tokens[i]);
    }

    // the model's response is actually at generated_sequence + num_input_tokens
    int* generated_sequence = (int*)mallocCheck((num_input_tokens + max_gen_length) * sizeof(int));
    memcpy(generated_sequence, input_tokens, num_input_tokens * sizeof(int));

    // seq_len_itr accounts for the number of input tokens
    // i.e. put new tokens into this index due to 0-indexing
    int seq_len_itr = num_input_tokens;

    // if you want to include prompt in generated output text string
    // strcpy(output_text, input_sequence);

    for (int i = 0; i < max_gen_length; ++i) {
        int next_token;
        gpt2_generate_next_token(model, generated_sequence, seq_len_itr, &next_token);

        if (seq_len_itr < model->config.max_seq_len) {
            generated_sequence[seq_len_itr] = next_token;
            seq_len_itr += 1;
        }

    }

    // this should not be segfaulting if max_gen_length number of tokens were generated properly
    for (int i = num_input_tokens; i < num_input_tokens + max_gen_length; ++i) {
        // decode the token and append to output text
        const char* next_token_str = tokenizer_decode(&tokenizer, generated_sequence[i]);
        strcat(output_text, next_token_str);
    }

    free(input_tokens);
    free(generated_sequence);
    tokenizer_free(&tokenizer);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("next_token_gen - Usage: %s \"<input sequence>\"\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(time(NULL));

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    int enable_tf32 = 0;
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    GPT2 model;
    gpt2_build_from_checkpoint(&model, checkpoint_file);
    printf("+-----------------------+\n");
    printf("| GPT2 MODEL PARAMETERS |\n");
    printf("+-----------------------+----------------------------------------------------+\n");
    printf("| max_sequence_length T | %-50d |\n", model.config.max_seq_len);
    printf("| vocab_size V          | %-50d |\n", model.config.vocab_size);
    printf("| padded_vocab_size Vp  | %-50d |\n", model.config.padded_vocab_size);
    printf("| num_layers L          | %-50d |\n", model.config.num_layers);
    printf("| num_heads NH          | %-50d |\n", model.config.num_heads);
    printf("| channels C            | %-50d |\n", model.config.channels);
    printf("| num_parameters        | %-50zu |\n", model.num_parameters);
    printf("+-----------------------+----------------------------------------------------+\n");

    // max_gen_length DOES NOT account for the input sequence length
    // 500 runs out of memory
    int max_gen_length = 50;

    const char* input_sequence = argv[1];

    char output_text[4096] = {0}; 

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    gpt2_generate_text(&model, input_sequence, max_gen_length, output_text);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("+----------+\n");
    printf("| GPT2 RUN |\n");
    printf("+----------+----------------------------------------------------------------------------------------------------------------------------------+\n");
    printf("|\n| INPUT TEXT: %s\n", input_sequence);
    printf("|\n| GENERATED TEXT: %s\n", output_text);
    printf("|\n| TIME FOR INFERENCING: %f ms\n", time_elapsed_s * 1000);
    printf("|\n| # TOKENS/SEC: %.3f\n", max_gen_length / time_elapsed_s);
    printf("+---------------------------------------------------------------------------------------------------------------------------------------------+\n");

    log_generation_output(input_sequence, output_text);

    return EXIT_SUCCESS;
}
