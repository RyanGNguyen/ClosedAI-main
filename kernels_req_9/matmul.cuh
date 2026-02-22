#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include "../utils/cuda_utils.cuh"
#include <cuda_runtime.h>

// CUTLASS Headers
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/arch/mma.h"
#include "cutlass/util/device_memory.h"

// Required for Split-K
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/reduction/thread/reduction_operators.h"

// Type Definitions
using ElementInputA = float;
using ElementInputB = float;
using ElementOutput = float;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor; 
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;


// Default configuration (128x128x32)
using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>;
constexpr int NumStages = 3;

constexpr int AlignmentA = 1;
constexpr int AlignmentB = 1;
constexpr int AlignmentC = 1;
constexpr int EpilogueVectorWidth = 1;


// Default GEMM
using CutlassGemmDefault = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadBlockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        EpilogueVectorWidth,
        ElementAccumulator,
        ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA, 
    AlignmentB
>;

// Small matrix configuration
using ThreadBlockShapeSmall = cutlass::gemm::GemmShape<64, 64, 32>;
using WarpShapeSmall = cutlass::gemm::GemmShape<32, 32, 32>;

using CutlassGemmSmall = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadBlockShapeSmall,
    WarpShapeSmall,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        EpilogueVectorWidth,
        ElementAccumulator,
        ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB
>;

// Large matrix configuration
using ThreadBlockShapeLarge = cutlass::gemm::GemmShape<256, 128, 32>;
using WarpShapeLarge = cutlass::gemm::GemmShape<64, 64, 32>;

using CutlassGemmLarge = cutlass::gemm::device::Gemm<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadBlockShapeLarge,
    WarpShapeLarge,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        EpilogueVectorWidth,
        ElementAccumulator,
        ElementCompute
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    NumStages,
    AlignmentA,
    AlignmentB
>;


using CutlassGemmSplitK = cutlass::gemm::device::GemmSplitKParallel<
    ElementInputA, LayoutA,
    ElementInputB, LayoutB,
    ElementOutput, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    ThreadBlockShape,
    WarpShape,
    InstructionShape,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        EpilogueVectorWidth,
        ElementAccumulator,
        ElementCompute
    >,
    cutlass::epilogue::thread::Convert<
        ElementOutput, 
        EpilogueVectorWidth,
        ElementAccumulator 
    >,
    cutlass::reduction::thread::ReduceAdd<
        ElementAccumulator, 
        ElementOutput, 
        EpilogueVectorWidth
    >,
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
    NumStages,
    AlignmentA,
    AlignmentB
>;

// Shared memory bias kernel (Safe for Alignment 1)
__global__ void add_bias_kernel_shared(
    float* __restrict__ out,
    const float* __restrict__ bias,
    const int M,
    const int N
) {
    extern __shared__ float s_bias[];
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        s_bias[i] = bias[i];
    }
    __syncthreads();
    
    const int row = blockIdx.x;
    if (row < M) {
        for (int col = threadIdx.x; col < N; col += blockDim.x) {
            out[row * N + col] += s_bias[col];
        }
    }
}

__global__ void add_bias_kernel(
    float* __restrict__ out,
    const float* __restrict__ bias,
    const int M,
    const int N
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M * N) {
        out[idx] += bias[idx % N];
    }
}


// CUTLASS Execution Helper
template<typename GemmKernel>
inline cutlass::Status run_gemm(
    float* out,
    const float* inp,
    const float* weight,
    int M, int N, int K,
    int split_k_slices = 1
) {
    const int lda = K;
    const int ldb = K;
    const int ldc = N;
    
    typename GemmKernel::Arguments args(
        {M, N, K},
        {inp, lda},
        {weight, ldb},
        {out, ldc},
        {out, ldc},
        {1.0f, 0.0f},
        split_k_slices
    );
    
    GemmKernel gemm;
    cutlass::Status status = gemm.can_implement(args);
    if (status != cutlass::Status::kSuccess) return status;
    
    size_t workspace_size = GemmKernel::get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    
    status = gemm.initialize(args, workspace.get());
    if (status != cutlass::Status::kSuccess) return status;
    
    return gemm();
}

// Main Forward Function
void matmul_forward(float *out, const float *inp, const float *weight, const float *bias,
                    int B, int T, int C, int OC) {
    
    const int M = B * T;
    const int N = OC;
    const int K = C;
    
    cutlass::Status status;
    const size_t problem_size = static_cast<size_t>(M) * N;
    
    bool use_split_k = (M * N < 1024 * 128) && (K >= 1024);
    
    if (use_split_k) {
        int split_slices = 4;
        status = run_gemm<CutlassGemmSplitK>(out, inp, weight, M, N, K, split_slices);
    }
    else if (M <= 64 || N <= 64 || problem_size < 8192) {
        status = run_gemm<CutlassGemmSmall>(out, inp, weight, M, N, K);
        if (status != cutlass::Status::kSuccess) {
            status = run_gemm<CutlassGemmDefault>(out, inp, weight, M, N, K);
        }
    } else if (M >= 512 && N >= 512 && problem_size > 262144) {
        status = run_gemm<CutlassGemmLarge>(out, inp, weight, M, N, K);
        if (status != cutlass::Status::kSuccess) {
            status = run_gemm<CutlassGemmDefault>(out, inp, weight, M, N, K);
        }
    } else {
        status = run_gemm<CutlassGemmDefault>(out, inp, weight, M, N, K);
    }
    
    if (status != cutlass::Status::kSuccess) {
        printf("CUTLASS GEMM failed: M=%d, N=%d, K=%d\n", M, N, K);
        return;
    }
    
    // Add bias
    if (bias != NULL) {
        const int total = M * N;
        const int block_size = 256;
        int max_shared;
        cudaDeviceGetAttribute(&max_shared, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        
        if (N <= 4096 && N * sizeof(float) <= (size_t)max_shared) {
            add_bias_kernel_shared<<<M, block_size, N * sizeof(float)>>>(out, bias, M, N);
        } else {
            const int grid = (total + block_size - 1) / block_size;
            add_bias_kernel<<<grid, block_size>>>(out, bias, M, N);
        }
    }
}

#endif // __MATMUL_KERNEL_CUH__