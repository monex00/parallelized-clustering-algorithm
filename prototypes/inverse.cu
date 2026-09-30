#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Macro per il controllo degli errori CUDA
#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
    exit(EXIT_FAILURE); \
}

// Macro per il controllo degli errori cuBLAS
#define CUBLAS_CHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at line %d\n", __LINE__); \
    exit(EXIT_FAILURE); \
}

// Funzione per invertire una matrice su GPU
void invertMatrix(cublasHandle_t handle, float* d_matrix, float* d_invMatrix, int n) {
    int *d_pivotArray, *d_info;
    CUDA_CHECK(cudaMalloc((void**)&d_pivotArray, n * sizeof(int))); // Pivot array
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));          // Info array

    // Array di puntatori per gestire il batch
    float* d_matrixArray[1] = {d_matrix};
    float* d_invMatrixArray[1] = {d_invMatrix};

    float** d_matrixArrayDevPtr;
    float** d_invMatrixArrayDevPtr;

    CUDA_CHECK(cudaMalloc((void**)&d_matrixArrayDevPtr, sizeof(d_matrixArray)));
    CUDA_CHECK(cudaMalloc((void**)&d_invMatrixArrayDevPtr, sizeof(d_invMatrixArray)));

    CUDA_CHECK(cudaMemcpy(d_matrixArrayDevPtr, d_matrixArray, sizeof(d_matrixArray), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_invMatrixArrayDevPtr, d_invMatrixArray, sizeof(d_invMatrixArray), cudaMemcpyHostToDevice));

    // Decomposizione LU
    CUBLAS_CHECK(cublasSgetrfBatched(handle, n, d_matrixArrayDevPtr, n, d_pivotArray, d_info, 1));

    // Calcolo dell'inversa della matrice
    CUBLAS_CHECK(cublasSgetriBatched(handle, n, (const float**)d_matrixArrayDevPtr, n, d_pivotArray, d_invMatrixArrayDevPtr, n, d_info, 1));

    // Pulizia della memoria temporanea
    CUDA_CHECK(cudaFree(d_pivotArray));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_matrixArrayDevPtr));
    CUDA_CHECK(cudaFree(d_invMatrixArrayDevPtr));
}

int main() {
    const int n = 3; // Dimensione della matrice
    float h_matrix[n * n] = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };

    float h_invMatrix[n * n];

    // Allocazione memoria sulla GPU
    float *d_matrix, *d_invMatrix;
    CUDA_CHECK(cudaMalloc((void**)&d_matrix, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_invMatrix, n * n * sizeof(float)));

    // Copia della matrice dalla CPU alla GPU
    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Inizializzazione di cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Calcolo dell'inversa
    invertMatrix(handle, d_matrix, d_invMatrix, n);

    // Copia della matrice inversa dalla GPU alla CPU
    CUDA_CHECK(cudaMemcpy(h_invMatrix, d_invMatrix, n * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Stampa del risultato
    printf("Matrice inversa:\n");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("%f ", h_invMatrix[i * n + j]);
        }
        printf("\n");
    }

    // Pulizia
    CUDA_CHECK(cudaFree(d_matrix));
    CUDA_CHECK(cudaFree(d_invMatrix));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}
