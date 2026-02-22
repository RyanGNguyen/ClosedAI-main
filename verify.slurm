#!/bin/bash
#SBATCH --job-name="verify"
#SBATCH --output="verify_output.out"
#SBATCH --error="verify_errors.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda/12.4

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# run random kernel output verification
srun ./output_verification_rand

# run kernel output verification against gold reference
# srun ./test_gpt2_kernels

# run overall inference test
# srun ./test_gpt2

# =========================== Profiling ===========================
# To use profiling tools, uncomment the below commands, and make sure
# your constraint parameter is set to "projects,perf,nvperf"
# Please make sure to set it back to "projects" for standard testing.

# --Generate nsys profile information, output will be in test_gpt2_nsys_prof.out--
# srun nsys profile --stats=true ./test_gpt2 > test_gpt2_nsys_prof.out

# --Generate analysis_file.ncu-rep with a detailed profiling result---
# ncu --set full -f -o analysis_file ./test_gpt2 > test_gpt2.out