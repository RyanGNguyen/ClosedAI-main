#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "cpu_kernels/local_attention.cuh"

//! TODO: Include your local attention kernel here!
#include "kernels/local_attention.cuh"

#define cudaCheck(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

void read_input_from_file(const char* filename, int* batch_size, int* seq_length, int* num_heads, int* head_dim, float** input) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening input file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d\n%d\n%d\n%d", batch_size, seq_length, num_heads, head_dim);

    int B = *batch_size; int T = *seq_length;
    int NH = *num_heads; int HS = *head_dim;
    *input = (float*) malloc(B * T * 3 * NH * HS * sizeof(float));

    // read Q, K, V
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int nh = 0; nh < NH; ++nh) {
                for (int hs = 0; hs < HS; ++hs) {
                    int idx = b * T * 3 * NH * HS + t * 3 * NH * HS + 0 * NH * HS + nh * HS + hs;
                    fscanf(file, "%f", &((*input)[idx]));
                }
            }
        }
    }
    fgetc(file);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int nh = 0; nh < NH; ++nh) {
                for (int hs = 0; hs < HS; ++hs) {
                    int idx = b * T * 3 * NH * HS + t * 3 * NH * HS + 1 * NH * HS + nh * HS + hs;
                    fscanf(file, "%f", &((*input)[idx]));
                }
            }
        }
    }
    fgetc(file);
    for (int b = 0; b < B; ++b) {
        for (int t = 0; t < T; ++t) {
            for (int nh = 0; nh < NH; ++nh) {
                for (int hs = 0; hs < HS; ++hs) {
                    int idx = b * T * 3 * NH * HS + t * 3 * NH * HS + 2 * NH * HS + nh * HS + hs;
                    fscanf(file, "%f", &((*input)[idx]));
                }
            }
        }
    }
    fclose(file);
}

void read_output_from_file(const char* filename, int* batch_size, int* seq_length, int* num_heads, int* head_dim, float** outputs) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening input file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    fscanf(file, "%d\n%d\n%d\n%d", batch_size, seq_length, num_heads, head_dim);

    int tensor_size = (*batch_size) * (*seq_length) * (*num_heads) * (*head_dim);
    *outputs = (float*) malloc(tensor_size * sizeof(float));

    for (int i = 0; i < tensor_size; ++i) fscanf(file, "%f", &((*outputs)[i]));
    fgetc(file);
    fclose(file);
}

void compare_arrays(float* computed_outputs, float* expected_outputs, int size, const char* test_name) {
    int diff_count = 0;
    double max_relative_error = 0.0;
    double rmse = 0.0;
    int first_mismatch_idx = -1;
    const double ERROR_THRESHOLD = 0.1; // 10% relative error threshold
    
    for (int i = 0; i < size; ++i) {
        double abs_error = fabs(computed_outputs[i] - expected_outputs[i]);
        double rel_error = abs_error / (fabs(expected_outputs[i]) + 1e-6);
        rmse += abs_error * abs_error;
        
        if (rel_error > max_relative_error) {
            max_relative_error = rel_error;
        }
        
        if (rel_error > ERROR_THRESHOLD) {
            if (first_mismatch_idx == -1) {
                first_mismatch_idx = i;
            }
            ++diff_count;
        }
    }
    
    rmse = sqrt(rmse / size);

    if (diff_count > 0) {
        printf("+++++++++++++++ %s TEST FAILED with %d differing values (%.2f%%) +++++++++++++++\n", 
               test_name, diff_count, (100.0 * diff_count) / size);
        printf("Max Relative Error: %.6f, RMSE: %.6f\n", max_relative_error, rmse);
        
        if (first_mismatch_idx >= 0) {
            printf("First mismatch at index %d: Expected %.6f, Got %.6f\n", 
                  first_mismatch_idx, expected_outputs[first_mismatch_idx], computed_outputs[first_mismatch_idx]);
        }
    } else {
        printf("-------------------- Test Passed! (Max Relative Error: %.6f) --------------------\n", 
               max_relative_error);
    }
}

int main() {
    printf("============= Local Attention Skeleton Test Code (Inputs and Golden Solution Provided) =============\n");
    
    cudaCheck(cudaSetDevice(0));
    
    int num_examples = 1;

    for (int example_idx = 0; example_idx < num_examples; example_idx++) {
        int batch_size = 0;
        int seq_length = 0; 
        int num_heads = 0;
        int head_dim = 0;
        
        float *input_data = NULL;
        float *expected_outputs = NULL;

        char input_filepath[512];
        char output_filepath[512];
        //snprintf(input_filepath, sizeof(input_filepath), "/work/hdd/bche/Project_GPT/local_attn_examples/example_%i.tcin", example_idx);
        //snprintf(output_filepath, sizeof(output_filepath), "/work/hdd/bche/Project_GPT/local_attn_examples/example_%i.tcout", example_idx);
        snprintf(input_filepath, sizeof(input_filepath), "/example_%i.tcin", example_idx);
        snprintf(output_filepath, sizeof(output_filepath), "/example_%i.tcout", example_idx);
        
        printf("\n Testing Example %d: \n", example_idx);
        
        // read provided input
        read_input_from_file(input_filepath, &batch_size, &seq_length, &num_heads, &head_dim, &input_data);
        // read golden solution values
        read_output_from_file(output_filepath, &batch_size, &seq_length, &num_heads, &head_dim, &expected_outputs);

        // --------------- CPU reference implementation test example ---------------
        
        int size = batch_size * seq_length * num_heads * head_dim;
        
        // separate copy of input for CPU to avoid input memory being overwritten
        float* att_inp_cpu = (float*) malloc(size * 3 * sizeof(float)); 
        float* qkvr_local = (float*) malloc(size * 3 * sizeof(float)); 
        float* att_out = (float*) malloc(size * sizeof(float));

        memcpy(att_inp_cpu, input_data, size * 3 * sizeof(float));

        // calls the CPU implementation to compute windowed attention outputs
        local_attention_forward_cpu(att_out, qkvr_local, att_inp_cpu, batch_size, seq_length, num_heads, head_dim);    
        
        //! TODO: uncomment the below line to see how to compare outputs of the provided CPU code with golden solution
        compare_arrays(att_out, expected_outputs, size, input_filepath);

        free(att_inp_cpu);
        free(qkvr_local);
        free(att_out);
        
        //! TODO: Modify the below code to test your GPU implementation!

        // --------------- Your GPU implementation test ---------------
        
        int tensor_size = batch_size * seq_length * num_heads * head_dim;

        float *device_input;         // queries, keys, values on device
        float *device_scratch;       // scratch space for intermediate calculations
        float *device_output;
        
        cudaCheck(cudaMalloc((void**)&device_input, tensor_size * 3 * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&device_scratch, tensor_size * 3 * sizeof(float)));
        cudaCheck(cudaMalloc((void**)&device_output, tensor_size * sizeof(float)));
        
        cudaCheck(cudaMemcpy(device_input, input_data, tensor_size * 3 * sizeof(float), 
                            cudaMemcpyHostToDevice));
        
        //! TODO: Call your GPU implementation here
        local_attention_forward(device_output, device_scratch, device_input, 
                          batch_size, seq_length, num_heads, head_dim);
        
        cudaCheck(cudaGetLastError());
        
        // Copy results back to host for comparison
        float* host_outputs = (float*) malloc(tensor_size * sizeof(float));
        cudaCheck(cudaMemcpy(host_outputs, device_output, tensor_size * sizeof(float), 
                            cudaMemcpyDeviceToHost));
        
        compare_arrays(host_outputs, expected_outputs, tensor_size, input_filepath);
        
        cudaCheck(cudaFree(device_input));
        cudaCheck(cudaFree(device_scratch));
        cudaCheck(cudaFree(device_output));
        
        free(host_outputs);

        free(input_data);
        free(expected_outputs);
    }
    
    printf("\n================== End of Tests Reached ==================\n");
    return 0;
}