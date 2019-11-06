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
 
#run nvprof --print-gpu-trace -m tensor_precision_fu_utilization ./real-wmma
srun nvprof --export-profile real-wmma.prof ./test-16real
