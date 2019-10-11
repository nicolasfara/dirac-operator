#!/bin/bash
#SBATCH --job-name=gpu-test
#SBATCH --error=gpu-test-%j.err
#SBATCH --output=gpu-test-%j.out
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=shortvolta
#SBATCH --gres=gpu:1
 
module load cuda/10.0
module load openmpi/4.0.1-cuda10.0
 
srun ./cublas-streams $@
