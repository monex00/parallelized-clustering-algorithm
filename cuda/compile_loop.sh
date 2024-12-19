#!/bin/bash
#SBATCH --job-name=mstep_stride
#SBATCH --output=results/cuda-stride-mstep-parallel_output.txt
#SBATCH --error=results/cuda-stride-mstep-parallel_error.txt
#SBATCH --partition=cascadelake
#SBATCH --gres=gpu:1           # Richiede 1 GPU

if [ ! -d "./build" ]; then
    mkdir build
fi

spack load cuda

# Compilazione
srun nvcc gmm-cuda-stride-reduction.cu -o ./build/gmm-cuda-stride-reduction -lcudart -lcublas -lm
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

# Esecuzione per valori del parametro crescenti
for param in $(seq 1500 40 4500); do
    echo "Esecuzione con parametro: $param" >> results/cuda-stride-mstep-parallel_output.txt
    srun ./build/gmm-cuda-stride-reduction $param >> results/cuda-stride-mstep-parallel_output.txt 2>> results/cuda-stride-mstep-parallel_error.txt
    if [ $? -ne 0 ]; then
        echo "Errore durante l'esecuzione con parametro: $param" >> results/cuda-stride-mstep-parallel_error.txt
    fi
done
