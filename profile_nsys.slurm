#!/bin/bash
#SBATCH --job-name="nsys_profile"
#SBATCH --output="nsys_profile.out"
#SBATCH --error="nsys_profile.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=16G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 01:00:00

module reset
module load cuda

echo "======================================"
echo "Nsys Profiling for Attention Kernels"
echo "Job ID: $SLURM_JOBID"
echo "Running on: $(hostname)"
echo "======================================"

# Create output directory for profiling results
mkdir -p profiling_results

echo ""
echo "1. Profiling Classic Attention (kernels_req_3)..."
echo "--------------------------------------"
nsys profile --stats=true -o profiling_results/classic_attn ./profile_classic_attn > profiling_results/classic_attn_nsys.txt
echo "Classic attention profiling complete! Results saved to profiling_results/classic_attn.nsys-rep"

echo ""
echo "2. Profiling Flash Attention without Tensor Cores (kernels_req_4)..."
echo "--------------------------------------"
nsys profile --stats=true -o profiling_results/flash_attn_no_tc ./profile_flash_attn_no_tc > profiling_results/flash_attn_no_tc_nsys.txt
echo "Flash attention (no TC) profiling complete! Results saved to profiling_results/flash_attn_no_tc.nsys-rep"

echo ""
echo "3. Profiling Flash Attention with Tensor Cores (kernels)..."
echo "--------------------------------------"
nsys profile --stats=true -o profiling_results/flash_attn_tc ./profile_flash_attn_tc > profiling_results/flash_attn_tc_nsys.txt
echo "Flash attention (with TC) profiling complete! Results saved to profiling_results/flash_attn_tc.nsys-rep"

echo ""
echo "======================================"
echo "All Nsys profiling complete!"
echo "Results are in the profiling_results/ directory:"
echo "  - classic_attn.nsys-rep (binary)"
echo "  - classic_attn_nsys.txt (statistics)"
echo "  - flash_attn_no_tc.nsys-rep (binary)"
echo "  - flash_attn_no_tc_nsys.txt (statistics)"
echo "  - flash_attn_tc.nsys-rep (binary)"
echo "  - flash_attn_tc_nsys.txt (statistics)"
echo ""
echo "To view results, download .nsys-rep files and open with NVIDIA Nsight Systems UI"
echo "======================================"
