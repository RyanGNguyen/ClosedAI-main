//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#include "utils/utils.h"
#include "gpt2.cuh"

int main(int argc, char *argv[]) {

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    int enable_tf32 = 1;
    // enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("test-gpt2-kernels - TF32 Mode is turned %s\n", enable_tf32 ? "ON." : "OFF.");
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "/work/hdd/bche/Project_GPT/gpt2_124M.bin");

    // int C = model.config.channels;
    int V = model.config.vocab_size;
    int Vp = model.config.padded_vocab_size;
    int maxT = model.config.max_seq_len;
    // int L = model.config.num_layers;

    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopenCheck("/work/hdd/bche/Project_GPT/gpt2_inference_verif.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20240327) { printf("test-gpt2 - Bad magic state file\n"); exit(EXIT_FAILURE); }
    if (state_header[1] != 2) {
        fprintf(stderr, "test-gpt2 - Bad version in state file\n");
        exit(EXIT_FAILURE);
    }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    assert(0 <= T && T <= maxT);
    printf("test-gpt2 - [State]\n");
    printf("Batch Size: %d\n", B);
    printf("Sequence Length: %d\n", T);

    // inputs and expected outputs, only used for error checking
    int* x = (int*)mallocCheck(B * T * sizeof(int));
    float* expected_logits = (float*) mallocCheck(B * T * V * sizeof(float));

    // read reference information
    freadCheck(x, sizeof(int), B*T, state_file);
    // freadCheck(y, sizeof(int), B*T, state_file);
	fseekCheck(state_file, B*T*sizeof(int), SEEK_CUR);
    freadCheck(expected_logits, sizeof(float), B*T*V, state_file);
    // freadCheck(expected_loss, sizeof(float), 1, state_file);
    // freadCheck(expected_grads_memory, sizeof(float), model.num_parameters, state_file);
    fcloseCheck(state_file);

    // overall OK signal for the test
    int allok = 1;

    // First, do target-free forward pass to validate logits
    gpt2_forward(&model, x, B, T);
    // at this point, target should be equal to expected_logits, let's compare
    // copy logits to CPU so we can compare them
    float* logits_cpu = (float*)mallocCheck(B * T * Vp * sizeof(float));
    cudaCheck(cudaMemcpy(logits_cpu, model.acts.output, B * T * Vp * sizeof(float), cudaMemcpyDeviceToHost));

    // compare the output logits from the forward pass
    // also careful that we don't access and compare the padded columns of logits
    int logits_ok = 1;
    double maxerr = 0.0f;
    double rmse = 0;
	printf("test-gpt2 - Printing 10 Expected vs Actual Logits: \n");
    for (int bt = 0; bt < B*T; bt++) {
        for (int v = 0; v < V; v++) {
            int i = bt * Vp + v; // linearized index
            float expected = expected_logits[bt*V + v];
            if (i < 10) {
                printf("%f, %f\n", expected, logits_cpu[i]);
            }
            if (isnan(logits_cpu[i])) {
                printf("=========================MISMATCH AT INDEX %d,%d: ", bt, v);
                printf("%f %f (logit is nan)=========================\n", expected, logits_cpu[i]);
                logits_ok = 0;
                bt = B*T; // to break out of both loops
                break;
            }
            double abserr = abs(logits_cpu[i] - expected);
            double relerr = abs(logits_cpu[i] - expected) / fmaxf(1e-3, abs(expected));
            double err = fmin(abserr, relerr);
            maxerr = fmaxf(maxerr, err);
            rmse += err * err;
            if (err > 0.1) {
                printf("=========================MISMATCH AT INDEX %d,%d: ", bt, v);
                printf("%f %f (err=%f)=========================\n", expected, logits_cpu[i], err);
                logits_ok = 0;
                bt = B*T; // to break out of both loops
                break;
            }
        }
    }
    rmse = sqrt(rmse / (B*T*V));
    if (rmse > 0.004) {
        printf("================ RMSE is too high! (max err %f, RMSE %f) ==================\n", maxerr, rmse);
        logits_ok = 0;
    }
    allok = allok && logits_ok;
    if(!logits_ok) { 
        printf("================ Logits NOT OK! ================\n"); 
    }
    else {
        printf("---------------- Logits OK! (max err %f, RMSE %f) ----------------\n", maxerr, rmse);
    }

    // final approval
    if (allok) {
        printf("_________________________ All tests passed! _________________________\n");
    } else {
        printf("_________________________ Some tests failed! _________________________\n");
    }

    // free everything
    free(x);
    free(expected_logits);
    gpt2_free(&model);
    cublasCheck(cublasDestroy(cublas_handle));

    return 0;
}