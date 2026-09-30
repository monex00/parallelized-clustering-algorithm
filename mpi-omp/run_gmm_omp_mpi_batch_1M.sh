#!/bin/bash
#SBATCH --job-name=gmm_omp_mpi_batch
#SBATCH --output=results/final/run_gmm_omp_mpi_batch_output1M10Ft5K.txt
#SBATCH --error=results/final/run_gmm_omp_mpi_batch_error1M10Ft5K.txt
#SBATCH --partition=broadwell
#SBATCH --nodes=12     

spack load eigen

if [ ! -d "./build" ]; then
    mkdir build
fi

# Compilazione
echo "Compilazione del programma MPI..."
mpic++ -fopenmp -I$(spack location -i eigen)/include gmm_mpi_omp.cpp -o ./build/gmm_mpi_omp1M
if [ $? -ne 0 ]; then
    echo "Compilazione fallita"
    exit 1
fi

## Configurazioni
CONFIGURATIONS=(
#  # 1 Nodo
"1 1 1"   # 1 nodo, 1 processo, 1 thread
  #"1 1 36"   # 1 nodo, 1 processo, 36 thread
"1 2 18"   # 1 nodo, 2 processi, 18 thread
"1 4 9"    # 1 nodo, 4 processi, 9 thread
"1 8 4"    # 1 nodo, 8 processi, 4 thread
"1 12 3"   # 1 nodo, 12 processi, 3 thread
"1 36 1"   # 1 nodo, 36 processi, 1 thread
##
# # # 2 Nodi
# # #"2 2 36"   # 2 nodi, 2 processi, 36 thread per nodo
"2 4 18"   # 2 nodi, 4 processi, 18 thread per nodo
"2 8 9"    # 2 nodi, 8 processi, 9 thread per nodo
"2 16 4"   # 2 nodi, 16 processi, 4 thread per nodo
"2 24 3"   # 2 nodi, 24 processi, 3 thread per nodo
"2 72 1"   # 2 nodi, 72 processi, 1 thread per processo
#
#  # 3 Nodi
#  #"3 3 36"   # 3 nodi, 3 processi, 36 thread per nodo
 "3 6 18"   # 3 nodi, 6 processi, 18 thread per nodo
 "3 12 9"   # 3 nodi, 12 processi, 9 thread per nodo
 "3 24 4"   # 3 nodi, 24 processi, 4 thread per nodo
 "3 36 3"   # 3 nodi, 36 processi, 3 thread per nodo
 "3 108 1"  # 3 nodi, 108 processi, 1 thread per processo
#
#  # 4 Nodi
#  #"4 4 36"   # 4 nodi, 4 processi, 36 thread per nodo
 "4 8 18"   # 4 nodi, 8 processi, 18 thread per nodo
 "4 16 9"   # 4 nodi, 16 processi, 9 thread per nodo
 "4 32 4"   # 4 nodi, 32 processi, 4 thread per nodo
 "4 48 3"   # 4 nodi, 48 processi, 3 thread per nodo
 "4 144 1"  # 4 nodi, 144 processi, 1 thread per processo
#
#  # 5 Nodi
  #"5 5 36"   # 5 nodi, 5 processi, 36 thread per nodo
 "5 10 18"  # 5 nodi, 10 processi, 18 thread per nodo
 "5 20 9"   # 5 nodi, 20 processi, 9 thread per nodo
 "5 40 4"   # 5 nodi, 40 processi, 4 thread per nodo
 "5 60 3"   # 5 nodi, 60 processi, 3 thread per nodo
 "5 180 1"  # 5 nodi, 180 processi, 1 thread per processo
#
#  # 6 Nodi
#  #"6 6 36"   # 6 nodi, 6 processi, 36 thread per nodo
"6 12 18"  # 6 nodi, 12 processi, 18 thread per nodo
"6 24 9"   # 6 nodi, 24 processi, 9 thread per nodo
"6 48 4"   # 6 nodi, 48 processi, 4 thread per nodo
"6 72 3"   # 6 nodi, 72 processi, 3 thread per nodo
"6 216 1"  # 6 nodi, 216 processi, 1 thread per processo
#
#    # 7 Nodi
#    #"7 7 36"   # 7 nodi, 7 processi, 36 thread per nodo
    "7 14 18"  # 7 nodi, 14 processi, 18 thread per nodo
    "7 28 9"   # 7 nodi, 28 processi, 9 thread per nodo
    "7 56 4"   # 7 nodi, 56 processi, 4 thread per nodo
    "7 84 3"   # 7 nodi, 84 processi, 3 thread per nodo
    "7 252 1"  # 7 nodi, 252 processi, 1 thread per processo
#
#    # 8 Nodi
#    #"8 8 36"   # 8 nodi, 8 processi, 36 thread per nodo
   "8 16 18"  # 8 nodi, 16 processi, 18 thread per nodo
   "8 32 9"   # 8 nodi, 32 processi, 9 thread per nodo
   "8 64 4"   # 8 nodi, 64 processi, 4 thread per nodo
   "8 96 3"   # 8 nodi, 96 processi, 3 thread per nodo
   "8 288 1"  # 8 nodi, 288 processi, 1 thread per processo
#
#    # 9 Nodi
#    #"9 9 36"   # 9 nodi, 9 processi, 36 thread per nodo
    "9 18 18"  # 9 nodi, 18 processi, 18 thread per nodo
    "9 36 9"   # 9 nodi, 36 processi, 9 thread per nodo
    "9 72 4"   # 9 nodi, 72 processi, 4 thread per nodo
    "9 108 3"  # 9 nodi, 108 processi, 3 thread per nodo
    "9 324 1"  # 9 nodi, 324 processi, 1 thread per processo
#
#    # 10 Nodi
    #"10 10 36"   # 10 nodi, 10 processi, 36 thread per nodo
    "10 20 18"   # 10 nodi, 20 processi, 18 thread per nodo
    "10 40 9"    # 10 nodi, 40 processi, 9 thread per nodo
    "10 80 4"    # 10 nodi, 80 processi, 4 thread per nodo
    "10 120 3"   # 10 nodi, 120 processi, 3 thread per nodo
    "10 360 1"   # 10 nodi, 360 processi, 1 thread per processo
#
#    # 11 Nodi
#    #"11 11 36"   # 11 nodi, 11 processi, 36 thread per nodo
    "11 22 18"   # 11 nodi, 22 processi, 18 thread per nodo
    "11 44 9"    # 11 nodi, 44 processi, 9 thread per nodo
    "11 88 4"    # 11 nodi, 88 processi, 4 thread per nodo
    "11 132 3"   # 11 nodi, 132 processi, 3 thread per nodo
    "11 396 1"   # 11 nodi, 396 processi, 1 thread per processo
#
#    # 12 Nodi
#    #"12 12 36"   # 12 nodi, 12 processi, 36 thread per nodo
   "12 24 18"   # 12 nodi, 24 processi, 18 thread per nodo
   "12 48 9"    # 12 nodi, 48 processi, 9 thread per nodo
   "12 96 4"    # 12 nodi, 96 processi, 4 thread per nodo
   "12 144 3"   # 12 nodi, 144 processi, 3 thread per nodo
   "12 432 1"   # 12 nodi, 432 processi, 1 thread per processo
)

#CONFIGURATIONS=(
   #    # 12 Nodi
#"12 12 36"   # 12 nodi, 12 processi, 36 thread per nodo
#"12 24 18"   # 12 nodi, 24 processi, 18 thread per nodo
#"12 48 9"    # 12 nodi, 48 processi, 9 thread per nodo
#"12 96 4"    # 12 nodi, 96 processi, 4 thread per nodo
#"12 144 3"   # 12 nodi, 144 processi, 3 thread per nodo
  # 1 Nodo
  #"1 1 1"   # 1 nodo, 1 processo, 1 thread
# "1 4 1"   # 1 nodo, 4 processi, 1 thread
# "1 8 1"   # 1 nodo, 8 processi, 1 thread
# "1 16 1"  # 1 nodo, 16 processi, 1 thread
# "1 36 1"  # 1 nodo, 36 processi, 1 thread

  # 2 Nodi
#  "2 42 1"   # 2 nodi, 4 processi, 1 thread per processo
#  "2 52 1"   # 2 nodi, 4 processi, 1 thread per processo
#  "2 62 1"   # 2 nodi, 4 processi, 1 thread per processo
#  "2 72 1"  # 2 nodi, 72 processi, 1 thread per processo
#
#  # 3 Nodi
#  "3 78 1"   # 3 nodi, 6 processi, 1 thread per processo
#  "3 88 1"   # 3 nodi, 6 processi, 1 thread per processo
#  "3 98 1"   # 3 nodi, 6 processi, 1 thread per processo
#  "3 108 1" # 3 nodi, 108 processi, 1 thread per processo
#
#  # 4 Nodi
#  "4 114 1"   # 4 nodi, 8 processi, 1 thread per processo
#  "4 124 1"   # 4 nodi, 8 processi, 1 thread per processo
#  "4 134 1"   # 4 nodi, 8 processi, 1 thread per processo
#  "4 144 1" # 4 nodi, 144 processi, 1 thread per processo
#)


NUM_RUNS=3  # Numero di esecuzioni per ogni configurazione

for CONFIG in "${CONFIGURATIONS[@]}"; do
    IFS=' ' read -r NUM_NODES NUM_TASKS CPUS_PER_TASK <<< "$CONFIG"
    echo "Esecuzione configurazione: $NUM_NODES nodi, $NUM_TASKS processi, $CPUS_PER_TASK thread"

    # Inizializza variabili
    TOTAL=0.0
    COUNT=0

    for ((i=1; i<=NUM_RUNS; i++)); do
        export OMP_NUM_THREADS=$CPUS_PER_TASK
        RESULT=$(srun --nodes=$NUM_NODES --ntasks=$NUM_TASKS --cpus-per-task=$CPUS_PER_TASK ./build/gmm_mpi_omp1M)
        if [[ ! $RESULT =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
            echo "Errore: Il programma non ha restituito un numero valido. Output: $RESULT"
            # metti result = total / count
            RESULT=$(echo "$TOTAL / $COUNT" | bc -l)
            #exit 1
        fi
        echo "Esecuzione $i Output: $RESULT"
        TOTAL=$(echo "$TOTAL + $RESULT" | bc)
        COUNT=$((COUNT + 1))
    done

    # Calcolo media
    if [ "$COUNT" -eq 0 ]; then
        echo "Nessun risultato raccolto. Controlla il programma."
        exit 1
    fi

    AVERAGE=$(echo "$TOTAL / $COUNT" | bc -l)

    # Formattazione dell'output con zero prima del punto decimale
    TOTAL=$(printf "%.6f" "$TOTAL")
    AVERAGE=$(printf "%.6f" "$AVERAGE")

    echo "Configurazione: $NUM_NODES nodi, $NUM_TASKS processi, $CPUS_PER_TASK thread"
    echo "Risultati totali: $TOTAL"
    echo "Numero di esecuzioni: $COUNT"
    echo "Media: $AVERAGE"
done

# Fine
exit 0
