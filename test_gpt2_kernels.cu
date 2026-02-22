//-----------------------------------------------------------------------------------------------
// Copyright (c) 2024 Andrej Karpathy
// Licensed under the MIT License. See the LICENSE file for details.
//
// Modifications Copyright (c) 2025 Hanwen Liu, Hrishi Shah, Kelin Zeng, Charles Pei, and Vijay Daita, ALL RIGHTS RESERVED.
//-----------------------------------------------------------------------------------------------

#include "utils/utils.h"
#include "gpt2.cuh"

void check_and_swap_acts(const char * kernel, FILE * kernel_state, float * output, size_t n) {
    cudaError_t lasterror = cudaGetLastError();
    if (lasterror != cudaSuccess) {
        printf("CUDA error running kernel %s: %s\n", kernel, cudaGetErrorString(lasterror));
        exit(EXIT_FAILURE);
    }

    float * output_cpu = (float*)mallocCheck(n * sizeof(float));
    cudaCheck(cudaMemcpy(output_cpu, output, n * sizeof(float), cudaMemcpyDeviceToHost));
    float * kernel_cpu = (float*)mallocCheck(n * sizeof(float));
    freadCheck(kernel_cpu, sizeof(float), n, kernel_state);
    int mismatches = 0;
    double maxerr = 0;
    double rmse = 0;
    for (size_t i = 0; i < n; i++) {
        if (isnan(output_cpu[i])) {
            mismatches++;
            continue;
        }
        double abserr = abs(output_cpu[i] - kernel_cpu[i]);
        double relerr = abs(output_cpu[i] - kernel_cpu[i]) / fmaxf(1e-3, abs(kernel_cpu[i]));
        double err = fmin(abserr, relerr);
        maxerr = fmax(maxerr, err);
        rmse += err * err;
        if (err > 0.05) { // 5% and 0.05 error threshold
            mismatches++;
            // printf("Mismatch at index %zu: %f vs %f\n", i, output_cpu[i], kernel_cpu[i]);
        }
    }
    rmse = sqrt(rmse/n);
    if (mismatches > 0) {
        printf("===================== %s KERNEL FAILED: %d mismatches (max err %f, RMSE %f) =====================\n", kernel, mismatches, maxerr, rmse);
    } else if (rmse > 0.001) {
        printf("===================== %s KERNEL FAILED: RMSE (max err %f, RMSE %f) =====================\n", kernel, maxerr, rmse);
    } else {
        printf("--------------------- %s Kernel Passed (max err %f, RMSE %f) ---------------------\n", kernel, maxerr, rmse);
    }
    // replace with expected output
    cudaCheck(cudaMemcpy(output, kernel_cpu, n * sizeof(float), cudaMemcpyHostToDevice));
    free(output_cpu);
    free(kernel_cpu);
}

int main(int argc, char *argv[]) {
    printf("Starting test_gpt2_kernels\n");

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);

    // setup cuBLAS and cuBLASLt
    cublasCheck(cublasCreate(&cublas_handle));
    int enable_tf32 = 0;
    // enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
    printf("test-gpt2-kernels - TF32 Mode is turned %s\n", enable_tf32 ? "ON." : "OFF.");
    cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
    cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
    cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

    // build the GPT-2 model from a checkpoint
    GPT2 modelval;
    GPT2 * model = &modelval;
    gpt2_build_from_checkpoint(model, "/work/hdd/bche/Project_GPT/gpt2_124M.bin");

    int C = model->config.channels;
    int V = model->config.vocab_size;
    int maxT = model->config.max_seq_len;
    int L = model->config.num_layers;
    int NH = model->config.num_heads;


    // load additional information that we will use for debugging and error checking
    FILE *state_file = fopenCheck("/work/hdd/bche/Project_GPT/gpt2_kernel_verif.bin", "rb");
    int state_header[256];
    freadCheck(state_header, sizeof(int), 256, state_file);
    if (state_header[0] != 20250211) { printf("test-gpt2-kernels - Bad magic kernel state file\n"); exit(EXIT_FAILURE); }
    if (state_header[1] != 1) {
        fprintf(stderr, "test-gpt2-kernels - Bad version in kernel state file\n");
        exit(EXIT_FAILURE);
    }
    int B = state_header[2]; // batch size, e.g. 4
    int T = state_header[3]; // time / sequence length (e.g. 64, up to maxT)
    assert(0 <= T && T <= maxT);

    // read inputs
    int* input_tokens = (int*)mallocCheck(B * T * sizeof(int));
    freadCheck(input_tokens, sizeof(int), B*T, state_file);

    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= input_tokens[i] && input_tokens[i] < V);
    }

    // allocate space for all the activations
	model->batch_size = B;
	model->seq_len = T;
	fill_in_activation_sizes(model->act_sizes, B, T, model->config);
	size_t num_activations = 0;
	for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
		num_activations += model->act_sizes[i];
	}
	model->num_activations = num_activations;
	model->acts_memory = malloc_and_point_activations(&model->acts, model->act_sizes);
	printf("test-gpt2-kernels - Allocated %zu MiB for activations\n", (num_activations * sizeof(float)) >> 20); // >> 20 is /(1024*1024)
	cudaCheck(cudaMalloc((void**)&model->inputs, B * T * sizeof(int)));
	cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * T * sizeof(float)));

    // copy inputs to the model
    cudaCheck(cudaMemcpy(model->inputs, input_tokens, B * T * sizeof(int), cudaMemcpyHostToDevice));

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    float* residual;
    encoder_forward(acts.encoded, model->inputs, params.wte, params.wpe, B, T, C); // encoding goes into residual[0]
    check_and_swap_acts("encoder", state_file, acts.encoded, B * T * C * sizeof(float));
    for (int l = 0; l < L; l++) {

        residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        float* l_ln1w = params.ln1w + l * C;
        float* l_ln1b = params.ln1b + l * C;
        float* l_qkvw = params.qkvw + l * 3*C * C;
        float* l_qkvb = params.qkvb + l * 3*C;
        float* l_attprojw = params.attprojw + l * C * C;
        float* l_attprojb = params.attprojb + l * C;
        float* l_ln2w = params.ln2w + l * C;
        float* l_ln2b = params.ln2b + l * C;
        float* l_fcw = params.fcw + l * 4*C * C;
        float* l_fcb = params.fcb + l * 4*C;
        float* l_fcprojw = params.fcprojw + l * C * 4*C;
        float* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        float* l_ln1 = acts.ln1 + l * B * T * C;
        float* l_ln1_mean = acts.ln1_mean + l * B * T;
        float* l_ln1_rstd = acts.ln1_rstd + l * B * T;
        float* l_qkvr = acts.qkvr + l * B * T * 3*C;
        float* l_atty = acts.atty + l * B * T * C;
        float* l_att = acts.att + l * B * NH * T * T;
        float* l_attproj = acts.attproj + l * B * T * C;
        float* l_residual2 = acts.residual2 + l * B * T * C;
        float* l_ln2 = acts.ln2 + l * B * T * C;
        float* l_ln2_mean = acts.ln2_mean + l * B * T;
        float* l_ln2_rstd = acts.ln2_rstd + l * B * T;
        float* l_fch = acts.fch + l * B * T * 4*C;
        float* l_fch_gelu = acts.fch_gelu + l * B * T * 4*C;
        float* l_fcproj = acts.fcproj + l * B * T * C;
        float* l_residual3 = acts.residual3 + l * B * T * C;
        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        float* scratch = acts.output;

        printf("\n______________________________ Layer %d ______________________________\n", l);

        // now do the forward pass
        layernorm_forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
        check_and_swap_acts("layernorm", state_file, l_ln1, B * T * C);
        matmul_forward(scratch, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);
        check_and_swap_acts("matmul", state_file, scratch, B * T * 3*C);
        attention_forward(l_atty, l_qkvr, l_att, scratch, B, T, C, NH);
        check_and_swap_acts("attention", state_file, l_atty, B * T * C);
        matmul_forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
        check_and_swap_acts("matmul", state_file, l_attproj, B * T * C);
        residual_forward(l_residual2, residual, l_attproj, B*T*C);
        check_and_swap_acts("residual", state_file, l_residual2, B * T * C);
        layernorm_forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);
        check_and_swap_acts("layernorm", state_file, l_ln2, B * T * C);
        matmul_forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
        check_and_swap_acts("matmul", state_file, l_fch, B * T * 4*C);
        gelu_forward(l_fch_gelu, l_fch, B*T*4*C);
        check_and_swap_acts("gelu", state_file, l_fch_gelu, B * T * 4*C);
        matmul_forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
        check_and_swap_acts("matmul", state_file, l_fcproj, B * T * C);
        residual_forward(l_residual3, l_residual2, l_fcproj, B*T*C);
        check_and_swap_acts("residual", state_file, l_residual3, B * T * C);

    }

    residual = acts.residual3 + (L-1) * B * T * C; // last residual is in residual3
    layernorm_forward(acts.lnf, acts.lnf_mean, acts.lnf_rstd, residual, params.lnfw, params.lnfb, B, T, C);
    check_and_swap_acts("layernorm", state_file, acts.lnf, B * T * C);

    printf("_______________________________ All kernel tests done! _______________________________\n");

    // close the file
    fcloseCheck(state_file);

    // free everything
    free(input_tokens);
    gpt2_free(model);
    cublasCheck(cublasDestroy(cublas_handle));

    return 0;
}