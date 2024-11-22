#!/bin/bash
#SBATCH --job-name=compile_and_run_matrix_stencil
#SBATCH --output=results/cuda_output.txt
#SBATCH --error=results/cuda_error.txt
#SBATCH --partition=cascadelake
#SBATCH --gres=gpu:1             # Richiede 1 GPU

if [ ! -d "./build" ]; then
    mkdir build
fi

spack load cuda

# Compilazione
srun nvcc gmm-cuda.cu -o ./build/gmm-cud -lcudart -lcublas -lm
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

# Esecuzione
srun ./build/gmm-cud