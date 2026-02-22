//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef GELU_CUH
#define GELU_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)

void gelu_forward_cpu(float* out, const float* inp, int N) {
    for (int i = 0; i < N; ++i) {
        float val = inp[i];
        float cube = 0.044715f * val * val * val;
        // Think about what GELU is, are we computing the exact gelu function here?
        out[i] = 0.5f * val * (1.0f + tanhf(GELU_SCALING_FACTOR * (val + cube)));
    }
}

#endif // GELU_CUH