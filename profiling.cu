#include "utils/cuda_utils.cuh"
#include "kernels/attention.cuh"
#include "kernels/matmul.cuh"
#include "kernels/softmax.cuh"
#include "kernels/gelu.cuh"
#include "kernels/layernorm.cuh"
#include <cstdio>
#include <cstring>
#include <stdio.h>
#include <iostream>
// Include CPU kernels
#include "cpu_kernels/attention.cuh"
#include "cpu_kernels/encoder.cuh"
#include "cpu_kernels/gelu.cuh"
#include "cpu_kernels/layernorm.cuh"
#include "cpu_kernels/matmul.cuh"
#include "cpu_kernels/residual.cuh"

#include <algorithm>
#include <string>
#include <stdexcept>
#include <cstdlib>
#include <cuda_profiler_api.h>
#include <type_traits>
#include "nvToolsExt.h"

using namespace std;

// Test kernel directly in profile.cu
__global__ void test_kernel(float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = 42.0f;
    }
}

void print_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

struct InputSize {
    int B;
    int T;
    int C;
    int OC;
};

inline int random_int() { return (rand() % 100); }

int main(int argc, char** argv) {
   InputSize sizes = {1, 16, 196, 64 };
   cublasCreate(&cublas_handle);
   size_t input_size = sizes.B * sizes.T * sizes.C * sizeof(float);
   size_t weight_size = sizes.C * sizes.OC * sizeof(float);
   size_t output_size = sizes.B * sizes.T *sizes.OC * sizeof(float);
   size_t bias_size= sizes.OC * sizeof(float);
   
   float* input = new float[sizes.B * sizes.T * sizes.C]; 
   float* weights = new float[sizes.OC * sizes.C]; 
   float* bias = new float[sizes.OC]; 
   float* output = (float*) malloc(output_size); 
   
   generate(weights, weights + sizes.C * sizes.OC, random_int);
   generate(input, input + sizes.B * sizes.T * sizes.C, random_int);
   generate(bias, bias + sizes.OC, random_int);
//    printf("-------------weights--------------\n");
//    print_matrix(weights, sizes.C, sizes.OC);
//    printf("-------------inputs--------------\n");
//    print_matrix(input, sizes.B * sizes.T, sizes.C);
   
    
   // Allowcate input
   float * output_device;
   float * weights_device;
   float * input_device;
   float * bias_device;
   cudaProfilerStart();
   
   nvtxRangePush("host-to-device");
   cudaMalloc((void**)&input_device, input_size);
   cudaMalloc((void**)&weights_device , weight_size);
   cudaMalloc((void**)&output_device, output_size);
   cudaMalloc((void**)&bias_device, bias_size);

   cudaMemcpy(input_device, input, input_size, cudaMemcpyHostToDevice);
   cudaMemcpy(weights_device, weights, weight_size, cudaMemcpyHostToDevice);
   cudaMemcpy(bias_device, bias, bias_size, cudaMemcpyHostToDevice);

   cudaError_t err = cudaGetLastError();
   if (err != cudaSuccess) {
       printf("CUDA error after memcpy: %s\n", cudaGetErrorString(err));
   }
   nvtxRangePop();
   
   // define grid and block sizes
   // launch kernel
   cout << "running cpu matmul" << endl;
   
   matmul_forward_cpu(
       output, 
       input, 
       weights, 
       bias, 
       sizes.B, 
       sizes.T, 
       sizes.C, 
       sizes.OC
   );
   
   
   print_matrix(output, sizes.B * sizes.T, sizes.OC);


   
   
   cout << "running tensor core matmul" << endl;

   // Clear any previous errors
   cudaGetLastError();

   nvtxRangePush("wmma_matmul");
   matmul_forward(
       output_device,
       input_device,
       weights_device,
       bias_device,
       sizes.B,
       sizes.T,
       sizes.C,
       sizes.OC
   );
   free(output);
   output = (float*) malloc(output_size);


   err = cudaGetLastError();
   if (err != cudaSuccess) {
       printf("CUDA error after synchronize: %s\n", cudaGetErrorString(err));
   }

   printf("Output before memcpy:\n");
   print_matrix(output, sizes.B*sizes.T, sizes.OC);
   cudaMemcpy(output, output_device, output_size, cudaMemcpyDeviceToHost);

   printf("Output after memcpy:\n");
   print_matrix(output, sizes.B*sizes.T, sizes.OC);
   nvtxRangePop();
   
   nvtxRangePush("tensor core matmul");
   cout << "running tensor core matmul" << endl;
   nvtxRangePop();
   
   nvtxRangePush("cublas matmul");
   cout << "running cuBLAS matmul" << endl;
   nvtxRangePop();
   cudaProfilerStop();
   
   // free mem
   
   cudaFree(input_device);
   cudaFree(output_device);
   cudaFree(weights_device);
   cudaFree(bias_device);
   delete[] input;
   delete[] weights;
   delete[] bias;
   free(output);
   cublasDestroy(cublas_handle);

   return 0; 
}