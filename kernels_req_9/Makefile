# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -arch=sm_86 -std=c++17 -rdc=true -g -lineinfo

# CUTLASS path - set this to your CUTLASS installation
CUTLASS_PATH ?= $(HOME)/cutlass
CUTLASS_INCLUDES = -I$(CUTLASS_PATH)/include -I$(CUTLASS_PATH)/tools/util/include

# Include and library paths
CUDA_INCLUDES = -I/usr/local/cuda/include
ALL_INCLUDES = $(CUDA_INCLUDES) $(CUTLASS_INCLUDES)
CUBLAS_LIBS = -L/usr/local/cuda/lib64 -lcublas -lcublasLt

# Kernel files
KERNELS = $(wildcard kernels/*.cuh)
CPU_KERNELS = $(wildcard cpu_kernels/*.cuh)

# Default target
all: output_verification_rand test_gpt2 test_gpt2_kernels next_token_generation

# Individual targets - now with CUTLASS includes
output_verification_rand: output_verification_rand.cu $(KERNELS) $(CPU_KERNELS)
	$(NVCC) $(CFLAGS) $(ALL_INCLUDES) -c output_verification_rand.cu -o output_verification_rand.o
	$(NVCC) $(CFLAGS) -o output_verification_rand output_verification_rand.o $(CUBLAS_LIBS)

test_gpt2: test_gpt2.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(ALL_INCLUDES) -c test_gpt2.cu -o test_gpt2.o
	$(NVCC) $(CFLAGS) -o test_gpt2 test_gpt2.o $(CUBLAS_LIBS)

test_gpt2_kernels: test_gpt2_kernels.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(ALL_INCLUDES) -c test_gpt2_kernels.cu -o test_gpt2_kernels.o
	$(NVCC) $(CFLAGS) -o test_gpt2_kernels test_gpt2_kernels.o $(CUBLAS_LIBS)

next_token_generation: next_token_generation.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(ALL_INCLUDES) -c next_token_generation.cu -o next_token_generation.o
	$(NVCC) $(CFLAGS) -o next_token_generation next_token_generation.o $(CUBLAS_LIBS)

# for milestone 3
local_attn_verify: local_attn_verify.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(ALL_INCLUDES) -g -c local_attn_verify.cu -o local_attn_verify.o
	$(NVCC) $(CFLAGS) -g -o local_attn_verify local_attn_verify.o $(CUBLAS_LIBS)

# Clean target
clean:
	rm -f *.o output_verification_rand test_gpt2 model_output next_token_generation local_attn_verify test_gpt2_kernels

.PHONY: all output_verification_rand model_output test_gpt2 test_gpt2_kernels next_token_generation local_attn_verify clean