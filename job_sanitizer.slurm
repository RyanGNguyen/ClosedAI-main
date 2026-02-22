#!/bin/bash
#SBATCH --job-name="sanitize gpt"
#SBATCH --output="sanitize.out"
#SBATCH --error="sanitize.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:020:00

module reset
# module unload cudatoolkit
module load cuda/12.8

set -e

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

SANITIZER_FLAGS="--show-backtrace no --print-limit 1 --error-exitcode -1 --destroy-on-device-error kernel"

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

srun echo -e "\nmemcheck tests:"
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS test_gpt2_kernels

srun echo -e "\ninitcheck tests:"
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS test_gpt2_kernels

srun echo -e "\nsynccheck tests:"
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS test_gpt2_kernels

sync

# Overall inference test

srun echo -e "\nmemcheck tests:"
srun compute-sanitizer --tool memcheck $SANITIZER_FLAGS test_gpt2

srun echo -e "\ninitcheck tests:"
srun compute-sanitizer --tool initcheck $SANITIZER_FLAGS test_gpt2

srun echo -e "\nsynccheck tests:"
srun compute-sanitizer --tool synccheck $SANITIZER_FLAGS test_gpt2

sync

srun echo -e "\nall tests completed."