//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#ifndef ATTENTION_CUH
#define ATTENTION_CUH

#include <assert.h>
#include <math.h>
#include <float.h>

void permute_kernel_cpu(float* q, float* k, float* v,
                        const float* inp,
                        int B, int T, int NH, int d) {
    for (int b = 0; b < B; ++b) {
        for (int nh_idx = 0; nh_idx < NH; ++nh_idx) {
            for (int n = 0; n < T; ++n) {
                for (int d_ = 0; d_ < d; ++d_) {
                    int inp_idx = (b * T * 3 * NH * d) + (n * 3 * NH * d) + (0 * NH * d) + (nh_idx * d) + d_;
                    q[b * NH * T * d + nh_idx * T * d + n * d + d_] = inp[inp_idx];
                    k[b * NH * T * d + nh_idx * T * d + n * d + d_] = inp[inp_idx + NH * d];
                    v[b * NH * T * d + nh_idx * T * d + n * d + d_] = inp[inp_idx + 2 * (NH * d)];
                }
            }
        }
    }
}

void unpermute_kernel_cpu(float* inp, float* out, int B, int T, int NH, int d) {
    for (int b = 0; b < B; ++b) {
        for (int n = 0; n < T; ++n) {
            for (int nh_idx = 0; nh_idx < NH; ++nh_idx) {
                for (int d_ = 0; d_ < d; ++d_) {
                    int other_idx = (b * NH * T * d) + (n * NH * d) + (nh_idx * d) + d_;
                    out[other_idx] = inp[b * NH * T * d + nh_idx * T * d + n * d + d_];
                }
            }
        }
    }
}


void softmax_forward_cpu(float* out, float inv_temperature, const float* inp, int N, int T) {
    for (int idx = 0; idx < N * T; ++idx) {
        // The thread's position in its current row
        int own_pos = idx % T;

        // Pointer to the current row
        const float* x = inp + idx * T;

        float maxval = -FLT_MAX;
        // Think about what this for loop condition is doing (Note the significance of i <= own_pos)
        for (int i = 0; i <= own_pos; ++i) {
            maxval = fmaxf(maxval, x[i]);
        }

        // Compute softmax
        float sumval = 0.0f;
        for (int i = 0; i <= own_pos; ++i) {
            float ev = expf(inv_temperature * (x[i] - maxval));
            sumval += ev;
            out[idx * T + i] = ev;
        }

        // Normalize
        float norm = 1.0f / sumval;
        for (int i = 0; i <= own_pos; ++i) {
            out[idx * T + i] *= norm;
        }
    }
}

void attention_forward_cpu(float* out, float* qkvr, float* att,
                           float* inp,
                           int B, int T, int C, int NH) {
    // Note: `inp` is re-used as a scratch buffer.
    // Its contents will be overwritten by this function.

    // inp is (B, T, 3C) QKV
    // preatt, att are (B, NH, T, T)
    // output is (B, T, C)
    int HS = C / NH; // head size

    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    float *q, *k, *v;
    q = qkvr + 0 * B * T * C;
    k = qkvr + 1 * B * T * C;
    v = qkvr + 2 * B * T * C;

    permute_kernel_cpu(q, k, v, inp, B, T, NH, HS);

    // Attention matmul: Q @ K^T
    // Compute pre-attention scores (B, NH, T, T)
    float* preatt = inp;
    for (int b = 0; b < B; b++) {
        for (int nh = 0; nh < NH; nh++) {
            for (int t1 = 0; t1 < T; t1++) {
                for (int t2 = 0; t2 < T; t2++) {
                    float sum = 0.0f;
                    for (int hs = 0; hs < HS; hs++) {
                       sum += k[b * NH * T * HS + nh * T * HS + t2 * HS + hs] * 
                               q[b * NH * T * HS + nh * T * HS + t1 * HS + hs];
                    }
                    preatt[b * NH * T * T + nh * T * T + t1 * T + t2] = sum;
                }
            }
        }
    }
    
    // Compute the softmax
    float scale = 1.0 / sqrtf(HS);
    softmax_forward_cpu(att, scale, preatt, B * NH, T);


    float* vaccum = inp;
    // Attention matmul: P @ V, where P holds the attention probabilities
    // (B, NH, T, T) @ (B, NH, T, HS) -> (B, NH, T, HS)
    for (int b = 0; b < B; ++b) {
        for (int nh = 0; nh < NH; ++nh) {
            for (int t = 0; t < T; ++t) {
                for (int hs = 0; hs < HS; ++hs) {
                    float sum = 0.0f;
                    for (int t2 = 0; t2 < T; ++t2) {
                        sum += att[b * NH * T * T + nh * T * T + t * T + t2] *
                            v[b * NH * T * HS + nh * T * HS + t2 * HS + hs];
                    }
                    vaccum[b * NH * T * HS + nh * T * HS + t * HS + hs] = sum;
                }
            }
        }
    }
  
    unpermute_kernel_cpu(vaccum, out, B, T, NH, HS);
}

#endif // ATTENTION_CUH