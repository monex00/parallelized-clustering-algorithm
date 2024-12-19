#!/bin/bash
#SBATCH --job-name=compile_and_run_matrix_stencil
#SBATCH --output=results/seq_output.txt
#SBATCH --error=results/seq_error.txt
#SBATCH --partition=broadwell

spack load eigen

if [ ! -d "./build" ]; then
    mkdir build
fi

srun g++ -O3 -I$(spack location -i eigen)/include gmm-seq.cpp -o ./build/gmm-seq -lm
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

# Esecuzione
srun ./build/gmm-seq 
