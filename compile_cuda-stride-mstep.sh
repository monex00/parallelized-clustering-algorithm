#!/bin/bash
#SBATCH --job-name=mstep_stride
#SBATCH --output=results/cuda-stride-mstep_output.txt
#SBATCH --error=results/cuda-stride-mstep_error.txt
#SBATCH --partition=cascadelake
#SBATCH --gres=gpu:1             # Richiede 1 GPU

if [ ! -d "./build" ]; then
    mkdir build
fi

spack load cuda

# Compilazione
srun nvcc gmm-cuda-stride-mstep.cu -o ./build/gmm-cuda-stride-mstep -lcudart -lcublas -lm
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

# Esecuzione
srun ./build/gmm-cuda-stride-mstep