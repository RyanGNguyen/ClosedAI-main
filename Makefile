# Compiler and flags
NVCC = nvcc
CFLAGS = -O3 -arch=sm_86 -std=c++11 -rdc=true -g -lineinfo

# Include and library paths for cuBLAS
CUBLAS_INCLUDES = -I/usr/local/cuda/include
CUBLAS_LIBS = -L/usr/local/cuda/lib64 -lcublas -lcublasLt  -lnvToolsExt

# Kernel files
KERNELS = $(wildcard kernels/*.cuh)
CPU_KERNELS = $(wildcard cpu_kernels/*.cuh)

# Default target
all: output_verification_rand test_gpt2 test_gpt2_kernels next_token_generation benchmark_attention


.profiling: profiling 
profiling: profiling.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -c profiling.cu -o profiling.o
	$(NVCC) $(CFLAGS) -o profiling profiling.o $(CUBLAS_LIBS)
	# ncu -o $(NCU_DIR)/matmul $(NCU_FLAGS)  profile --profile

# Individual targets
.output_verification_rand: output_verification_rand
output_verification_rand: output_verification_rand.cu $(KERNELS) $(CPU_KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -c output_verification_rand.cu -o output_verification_rand.o
	$(NVCC) $(CFLAGS) -o output_verification_rand output_verification_rand.o $(CUBLAS_LIBS)

.test_gpt2: test_gpt2
test_gpt2: test_gpt2.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -c test_gpt2.cu -o test_gpt2.o
	$(NVCC) $(CFLAGS) -o test_gpt2 test_gpt2.o $(CUBLAS_LIBS)

.test_gpt2_kernels: test_gpt2_kernels
test_gpt2_kernels: test_gpt2_kernels.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -c test_gpt2_kernels.cu -o test_gpt2_kernels.o
	$(NVCC) $(CFLAGS) -o test_gpt2_kernels test_gpt2_kernels.o $(CUBLAS_LIBS)

.next_token_generation: next_token_generation
next_token_generation: next_token_generation.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -c next_token_generation.cu -o next_token_generation.o
	$(NVCC) $(CFLAGS) -o next_token_generation next_token_generation.o $(CUBLAS_LIBS)

# for milestone 3
.local_attn_verify: local_attn_verify
local_attn_verify: local_attn_verify.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -g -c local_attn_verify.cu -o local_attn_verify.o
	$(NVCC) $(CFLAGS) -g -o local_attn_verify local_attn_verify.o $(CUBLAS_LIBS)

.benchmark_attention: benchmark_attention
benchmark_attention: kernels_req_5/benchmark_attention.cu $(KERNELS)
	$(NVCC) $(CFLAGS) $(CUBLAS_INCLUDES) -I. -c kernels_req_5/benchmark_attention.cu -o benchmark_attention.o
	$(NVCC) $(CFLAGS) -o benchmark_attention benchmark_attention.o $(CUBLAS_LIBS)

# Clean target
clean:
	rm -f *.o *.err *.out output_verification_rand test_gpt2 model_output next_token_generation local_attn_verify benchmark_attention


.PHONY: all output_verification_rand model_output test_gpt2 test_gpt2_kernels next_token_generation local_attn_verify benchmark_attention clean
