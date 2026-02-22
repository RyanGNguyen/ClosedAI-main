#!/bin/bash
#SBATCH --job-name="generate_tokens"
#SBATCH --output="generate_tokens_output.out"
#SBATCH --error="generate_tokens_errors.err"
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

# run model
# srun ./next_token_generation "The meaning of life is"


# =========================== Profiling ===========================
# To use profiling tools, uncomment the below commands, and make sure
# your constraint parameter is set to "projects,perf,nvperf"
# Please make sure to set it back to "projects" for standard testing.

# --Generate nsys profile information, output will be in test_gpt2_nsys_prof.out--
srun nsys profile --stats=true ./next_token_generation "The meaning of life is" > profile_nsys.out

# --Generate analysis_file.ncu-rep with a detailed profiling result---
ncu --set full -f -o analysis_file ./next_token_generation "The meaning of life is" > profile.out