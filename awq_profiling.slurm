#!/bin/bash
#SBATCH --job-name="awq_profiling"
#SBATCH --output="awq_profiling_output.out"
#SBATCH --error="awq_profiling_errors.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module unload cudatoolkit
module load cuda/12.8

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# =========================== AWQ Model Profiling ===========================
# This script profiles the AWQ model only (default behavior)
# To run comparison mode instead, add --compare flag to the command
# Make sure constraint parameter is set to "projects,perf,nvperf" for profiling
# Set it back to "projects" for standard testing.

# --Generate nsys profile information, output will be in awq_profiling_nsys.out--
srun nsys profile --stats=true ./awq_test "The meaning of life is" > profile_nsys.out

# --Generate awq_analysis.ncu-rep with a detailed profiling result---
ncu --set full -f -o awq_analysis ./awq_test "The meaning of life is" > profile.out

