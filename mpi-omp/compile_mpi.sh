#!/bin/bash
#SBATCH --job-name=gmm_mpi_omp            # Nome del job
#SBATCH --output=results/gmm_mpi_omp.out  # File di output
#SBATCH --error=results/gmm_mpi_omp.err   # File di errori
#SBATCH --partition=broadwell             # Partizione (verifica il nome corretto)
#SBATCH --nodes=2                         # Numero di nodi better 3
#SBATCH --ntasks=6                      # Numero totale di task MPI better 6 , 9 , 12 ,15
#SBATCH --cpus-per-task=9            # Numero di CPU per task MPI better 18 = ~25 con 12 = 24.2518 con 9 = 23.2676 con 7 = 23.3591
#SBATCH --time=02:00:00                   # Tempo massimo (aggiunto per evitare timeout)

#4          4 
#24         16
#6          9
#18.3301    18.5753

# Carica i moduli necessari (ad esempio, il compilatore e MPI)
#module load gcc                           # Carica il compilatore GCC (se necessario)
#module load openmpi                       # Carica OpenMPI (se necessario)

# Carica librerie con Spack

spack load eigen
#spack load intel-openapi-vtune


# Crea la directory di build se non esiste
#if [ ! -d "./vtuneres" ]; then
#    mkdir vtuneres
#fi

if [ ! -d "./build" ]; then
    mkdir build
fi

# Compila il programma
mpic++ -fopenmp \
    -I$(spack location -i eigen)/include \
    gmm_mpi_omp_old.cpp -o ./build/gmm_mpi_omp

# Verifica se la compilazione è riuscita
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

# Imposta il numero di thread OpenMP (uguale a --cpus-per-task)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Esegui il programma con srun
srun ./build/gmm_mpi_omp
#vtune -collect hotspots -result-dir ./vtuneres ./build/gmm_mpi_omp
