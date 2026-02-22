#ifndef LOCAL_ATTENTION_CUH
#define LOCAL_ATTENTION_CUH

#include <assert.h>
#include <math.h>
#include <float.h>
#include "attention.cuh"

#define WINDOW_SIZE 128

void local_attention_forward_cpu(float* out, float* qkvr, float* inp,
                                int B, int T, int NH, int HS) {

    float *q, *k, *v;
    q = qkvr + 0 * B * T * NH * HS;
    k = qkvr + 1 * B * T * NH * HS;
    v = qkvr + 2 * B * T * NH * HS;
    
    // permute and separate inp from (B, T, 3, NH, HS) to 3X (B, NH, T, HS)
    permute_kernel_cpu(q, k, v, inp, B, T, NH, HS);

    float scale = 1.0 / sqrtf(HS);
    float* preatt = inp;
    for(int b = 0; b < B; b++){
        for(int nh = 0; nh < NH; nh++){
            for(int q_t = 0; q_t < T; q_t++){
                // weights buffer for the current window
                float* weights = (float*) calloc(WINDOW_SIZE, sizeof(float));
                
                // causal window start for keys
                int k_start = max(0, q_t - WINDOW_SIZE);
                
                // raw attention scores within the window
                for(int k_t = k_start; k_t <= q_t; k_t++){
                    float sum = 0.0f;
                    for(int hs = 0; hs < HS; hs++){
                        sum += k[b * NH * T * HS + nh * T * HS + k_t * HS + hs] * 
                            q[b * NH * T * HS + nh * T * HS + q_t * HS + hs];
                    }
                    weights[q_t - k_t] = sum;
                }
                
                // softmax over the window
                float maxval = -FLT_MAX;
                for(int k_t = k_start; k_t <= q_t; k_t++){
                    maxval = fmaxf(maxval, weights[q_t - k_t]);
                }
                float sumval = 0.0f;
                for(int k_t = k_start; k_t <= q_t; k_t++){
                    float ev = expf(scale * (weights[q_t - k_t] - maxval));
                    sumval += ev;
                    weights[q_t - k_t] = ev;
                }

                for(int k_t = k_start; k_t <= q_t; k_t++){
                    weights[q_t - k_t] /= sumval;
                }

                // weighted sum of values
                for(int hs = 0; hs < HS; hs++){
                    float sum = 0;
                    for(int k_t = k_start; k_t <= q_t; k_t++){
                        sum += weights[q_t - k_t] * v[b * NH * T * HS + nh * T * HS + k_t * HS + hs];
                    }
                    preatt[b * NH * T * HS + nh * T * HS + q_t * HS + hs] = sum;
                }

                free(weights);
            }
        }
    }

    unpermute_kernel_cpu(preatt, out, B, T, NH, HS);
}

#endif // LOCAL_ATTENTION_CUH