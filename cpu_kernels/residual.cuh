//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef RESIDUAL_CUH
#define RESIDUAL_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void residual_forward_cpu(float* out, const float* inp1, const float* inp2, int N) {
    // Element-wise addition for residual connections
    for (int i = 0; i < N; ++i) {
        out[i] = inp1[i] + inp2[i];
    }
}

#endif // RESIDUAL_CUH