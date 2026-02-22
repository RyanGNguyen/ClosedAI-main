#!/bin/bash
#SBATCH --job-name="awq"
#SBATCH --output="awq.out"
#SBATCH --error="awq.err"
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


# run awq test 
# srun ./awq 50 


# =========================== Profiling ===========================
# To use profiling tools, uncomment the below commands, and make sure
# your constraint parameter is set to "projects,perf,nvperf"
# Please make sure to set it back to "projects" for standard testing.

# --Generate nsys profile information, output will be in test_gpt2_nsys_prof.out--
# srun nsys profile --stats=true ./test_gpt2 > test_gpt2_nsys_prof.out

# --Generate analysis_file.ncu-rep with a detailed profiling result---
# ncu --set full -f -o analysis_file ./test_gpt2 > test_gpt2.out