//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef __LOGITS_CUH__
#define __LOGITS_CUH__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>

/* ============================== SOFTMAX & PROBABILITY HELPERS ============================== */

void compute_softmax_cpu(const float* logits, float* probs, int vocab_size, float temperature = 1.0f) {
    // Find max logit for numerical stability
    float max_logit = -FLT_MAX;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Compute exp and sum (with temperature scaling)
    float sum_exp = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    
    // Normalize
    for (int i = 0; i < vocab_size; i++) {
        probs[i] /= sum_exp;
    }
}

// Sample next token from logits using temperature-based sampling
// Higher temperature = more randomness, lower = more deterministic
int sample_next_token_temperature(const float* logits, int vocab_size, float temperature) {
    // Compute softmax probabilities with temperature scaling
    float* probabilities = (float*)malloc(vocab_size * sizeof(float));
    compute_softmax_cpu(logits, probabilities, vocab_size, temperature);

    // Sample from distribution using cumulative probability
    float random_value = (float)rand() / RAND_MAX;
    float cumulative = 0.0f;

    for (int i = 0; i < vocab_size; i++) {
        cumulative += probabilities[i];
        if (random_value < cumulative) {
            free(probabilities);
            return i;
        }
    }

    free(probabilities);

    // default fallback (should rarely reach here)
    return vocab_size - 1;
}

/* ============================== PERPLEXITY CALCULATION ============================== */

// Calculate perplexity based on how well the model predicts actual reference tokens
// Lower perplexity = better model (assigns higher probability to ground truth)
//
// Inputs:
//   - logits: (num_tokens, vocab_size) model output logits
//   - target_tokens: (num_tokens) actual next token IDs to predict
//   - num_tokens: number of token positions to evaluate
//   - vocab_size: vocabulary size
//
// Returns: perplexity value (exp of average negative log-likelihood)
double calculate_perplexity(const float* logits, const int* target_tokens, 
                           int num_tokens, int vocab_size) {
    
    const double epsilon = 1e-10;  // For numerical stability
    double total_nll = 0.0;  // negative log-likelihood
    
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    
    // For each token position
    for (int t = 0; t < num_tokens; t++) {
        const float* logits_t = logits + t * vocab_size;
        int target_token = target_tokens[t];
        
        // Validate target token is in valid range
        if (target_token < 0 || target_token >= vocab_size) {
            fprintf(stderr, "Error: Invalid target token %d at position %d\n", 
                    target_token, t);
            free(probs);
            return -1.0;
        }
        
        // Compute softmax probabilities
        compute_softmax_cpu(logits_t, probs, vocab_size);
        
        // Get probability assigned to the actual target token
        double target_prob = fmax(probs[target_token], epsilon);
        
        // Accumulate negative log-likelihood
        total_nll -= log(target_prob);
    }
    
    free(probs);
    
    // Average NLL and convert to perplexity
    double avg_nll = total_nll / num_tokens;
    double perplexity = exp(avg_nll);
    
    return perplexity;
}

// Print perplexity result with formatted output
void print_perplexity(const char* model_name, double perplexity, int num_tokens) {
    printf("┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-63s │\n", model_name);
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ Tokens Evaluated    │ %-43d │\n", num_tokens);
    printf("│ Perplexity          │ %-43.6f │\n", perplexity);
    printf("└─────────────────────────────────────────────────────────────────┘\n");
    
    if (perplexity < 0) {
        printf("❌ ERROR: Invalid perplexity calculation\n");
    } else if (perplexity < 10.0) {
        printf("✅ EXCELLENT: Very low perplexity\n");
    } else if (perplexity < 30.0) {
        printf("✅ GOOD: Acceptable perplexity\n");
    } else if (perplexity < 100.0) {
        printf("⚠️  FAIR: Higher perplexity, model may struggle\n");
    } else {
        printf("❌ POOR: Very high perplexity, significant quality degradation\n");
    }
}

#endif // __LOGITS_CUH__
