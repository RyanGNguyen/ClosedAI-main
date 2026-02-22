#!/bin/bash
#SBATCH --job-name="profiling ncu"
#SBATCH --output="ncu.out"
#SBATCH --error="ncu.err"
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="scratch"
#SBATCH --constraint="projects,perf,nvperf"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00


module reset
module load cuda/12.4

python kernels_req_4/profile_attention_kernels.py