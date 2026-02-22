//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef LAYERNORM_CUH
#define LAYERNORM_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void layernorm_forward_cpu(float* out, float* mean, float* rstd, float* inp, float* weight, float* bias,
                           int B, int T, int C) {
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            // For each token in th batch, normalize across the channel dimension
            float sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                sum += inp[b * T * C + t * C + c];
            }
            float m = sum / C;

            sum = 0.0f;
            for (int c = 0; c < C; ++c) {
                float diff = inp[b * T * C + t * C + c] - m;
                sum += diff * diff;
            }
            // Standard deviation of the channel values
            // but think abuot why we take the reciprocal
            float s = rsqrtf(sum / C + 1e-5f);
            
            for (int c = 0; c < C; ++c) {
                float n = s * (inp[b * T * C + t * C + c] - m);
                out[b * T * C + t * C + c] = n * weight[c] + bias[c];
            }
        }
    }
}

#endif // LAYERNORM_CUH