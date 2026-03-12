//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
*/
#ifndef UTILS_TOKENIZER_H
#define UTILS_TOKENIZER_H

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
#include "utils.h"

// ----------------------------------------------------------------------------

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
} Tokenizer;

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        printf("---\n");
        printf("Tokenizer - WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("Tokenizer - Invalid token id %u!\n", token_id);
        return NULL;
    }
}

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

void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}

#endif  // UTILS_TOKENIZER_H
