//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef MATMUL_CUH
#define MATMUL_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void matmul_forward_cpu(float* out, const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC) {
    // Input (B,T,C) @ Weight (OC, C) + Bias is (OC) = Output (B,T,OC)
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* out_token_ptr = out + b * T * OC + t * OC;
            const float* inp_token_ptr = inp + b * T * C + t * C;
            for (int o = 0; o < OC; o++) {
                float acc = (bias != NULL) ? bias[o] : 0.0f;
                const float* weight_row_ptr = weight + o*C;
                // Dot product of the input token values and corresponding row of weight values
                for (int i = 0; i < C; i++) {
                    acc += inp_token_ptr[i] * weight_row_ptr[i];
                }
                out_token_ptr[o] = acc;
            }
        }
    }
}

#endif // MATMUL_CUH