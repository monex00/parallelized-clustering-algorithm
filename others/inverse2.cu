#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err) if (err != cudaSuccess) { printf("CUDA Error: %s\n", cudaGetErrorString(err)); return -1; }
#define CUBLAS_CHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS Error\n"); return -1; }

int main() {
    const int n = 3; // Dimensione di ogni matrice
    const int batchSize = 3; // Numero di matrici

    // Host input: 3 matrici (n x n)
    float h_matrices[batchSize][n][n] = {
        { {4, 2, 1}, {2, 5, 3}, {1, 3, 6} }, // Matrice 1
        { {3, 1, 2}, {1, 4, 1}, {2, 1, 3} }, // Matrice 2
        { {2, 1, 1}, {1, 3, 2}, {1, 2, 4} }  // Matrice 3
    };

    // Allocazione in GPU
    float* d_matrices;
    CUDA_CHECK(cudaMalloc((void**)&d_matrices, batchSize * n * n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_matrices, h_matrices, batchSize * n * n * sizeof(float), cudaMemcpyHostToDevice));

    // Array di puntatori GPU per batched operations
    float** d_matrixArray;
    CUDA_CHECK(cudaMalloc((void**)&d_matrixArray, batchSize * sizeof(float*)));
    for (int i = 0; i < batchSize; ++i) {
        float* matrixAddress = d_matrices + i * n * n; // Calcola indirizzo per ogni matrice
        CUDA_CHECK(cudaMemcpy(d_matrixArray + i, &matrixAddress, sizeof(float*), cudaMemcpyHostToDevice));
    }

    // Array di pivot e info
    int* d_pivotArray;
    int* d_infoArray;
    CUDA_CHECK(cudaMalloc((void**)&d_pivotArray, batchSize * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_infoArray, batchSize * sizeof(int)));

    // Handle di cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Fattorizzazione LU per batched matrices
    CUBLAS_CHECK(cublasSgetrfBatched(handle, n, d_matrixArray, n, d_pivotArray, d_infoArray, batchSize));

    // Calcolo dell'inversa per batched matrices
    CUBLAS_CHECK(cublasSgetriBatched(handle, n, (const float**)d_matrixArray, n, d_pivotArray, d_matrixArray, n, d_infoArray, batchSize));

    // Copia risultato su host
    float h_invMatrices[batchSize][n][n];
    CUDA_CHECK(cudaMemcpy(h_invMatrices, d_matrices, batchSize * n * n * sizeof(float), cudaMemcpyDeviceToHost));

    // Stampa delle matrici inverse
    for (int k = 0; k < batchSize; ++k) {
        printf("Inversa della matrice %d:\n", k + 1);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                printf("%f ", h_invMatrices[k][i][j]);
            }
            printf("\n");
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_matrices));
    CUDA_CHECK(cudaFree(d_matrixArray));
    CUDA_CHECK(cudaFree(d_pivotArray));
    CUDA_CHECK(cudaFree(d_infoArray));
    CUBLAS_CHECK(cublasDestroy(handle));

    return 0;
}