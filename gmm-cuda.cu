#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

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

__global__ void computeResponsibilities(
    const float* data, const float* means, const float* invCovMatrices,
    const float* determinants, const float* weights,
    float* responsibilities, int n, int d, int k, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    for (int cluster = 0; cluster < k; ++cluster) {
        float mahalanobis = 0.0;
        for (int i = 0; i < d; ++i) {
            float diff = data[idx * d + i] - means[cluster * d + i];
            for (int j = 0; j < d; ++j) {
                mahalanobis += diff * invCovMatrices[cluster * d * d + i * d + j] * diff;
            }
        }

        float likelihood = expf(-0.5 * mahalanobis) / sqrtf(powf(2 * M_PI, d) * determinants[cluster]);
        responsibilities[idx * k + cluster] = weights[cluster] * likelihood;
    }
}

__global__ void normalizeResponsibilities(float* responsibilities, int k, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float sum = 0.0;
    for (int cluster = 0; cluster < k; ++cluster) {
        sum += responsibilities[idx * k + cluster];
    }
    for (int cluster = 0; cluster < k; ++cluster) {
        responsibilities[idx * k + cluster] /= sum;
    }
}

__global__ void mStep(
    const float* data, const float* responsibilities, float* means,
    float* covMatrices, float* weights, int n, int d, int k, int N) {

    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) return;

    float weightSum = 0.0;

    // Update means
    for (int i = 0; i < d; ++i) {
        means[cluster * d + i] = 0.0;
    }

    for (int idx = 0; idx < N; ++idx) {
        float r = responsibilities[idx * k + cluster];
        weightSum += r;
        for (int i = 0; i < d; ++i) {
            means[cluster * d + i] += r * data[idx * d + i];
        }
    }

    for (int i = 0; i < d; ++i) {
        means[cluster * d + i] /= weightSum;
    }

    // Update covariance matrices
    for (int i = 0; i < d * d; ++i) {
        covMatrices[cluster * d * d + i] = 0.0;
    }

    for (int idx = 0; idx < N; ++idx) {
        float r = responsibilities[idx * k + cluster];
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                float diff_i = data[idx * d + i] - means[cluster * d + i];
                float diff_j = data[idx * d + j] - means[cluster * d + j];
                covMatrices[cluster * d * d + i * d + j] += r * diff_i * diff_j;
            }
        }
    }

    for (int i = 0; i < d * d; ++i) {
        covMatrices[cluster * d * d + i] /= weightSum;
    }

    // Update weights
    weights[cluster] = weightSum / N;
}

void computeInverseMatrices(
    cublasHandle_t handle, float* d_matrices, int n, int batchSize,
    float* d_invMatrices) {

    // Array di puntatori per operazioni batched
    float** d_matrixArray;
    CUDA_CHECK(cudaMalloc((void**)&d_matrixArray, batchSize * sizeof(float*)));
    float** d_invMatrixArray;
    CUDA_CHECK(cudaMalloc((void**)&d_invMatrixArray, batchSize * sizeof(float*)));


    for (int i = 0; i < batchSize; ++i) {
        float* matrixAddress = d_matrices + i * n * n;
        float* invMatrixAddress = d_invMatrices + i * n * n;

        CUDA_CHECK(cudaMemcpy(d_matrixArray + i, &matrixAddress, sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_invMatrixArray + i, &invMatrixAddress, sizeof(float*), cudaMemcpyHostToDevice));
    }

    // Array di pivot e info
    int* d_pivotArray;
    int* d_infoArray;
    CUDA_CHECK(cudaMalloc((void**)&d_pivotArray, batchSize * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_infoArray, batchSize * sizeof(int)));

    // LU decomposition
    CUBLAS_CHECK(cublasSgetrfBatched(handle, n, d_matrixArray, n, d_pivotArray, d_infoArray, batchSize));

    // Inversion
    CUBLAS_CHECK(cublasSgetriBatched(handle, n, (const float**)d_matrixArray, n, d_pivotArray, d_invMatrixArray, n, d_infoArray, batchSize));

    // Cleanup
    cudaFree(d_matrixArray);
    cudaFree(d_invMatrixArray);
    cudaFree(d_pivotArray);
    cudaFree(d_infoArray);
}

/* int main() {
    const int d = 2;       // Dimensione delle feature
    const int k = 2;       // Numero di cluster
    const int N = 1000;    // Numero di campioni
    const int maxIter = 10;

    // Allocazione dati sintetici su host (random per esempio)
    float h_data[N * d];
    for (int i = 0; i < N * d; ++i) h_data[i] = (float)(rand() % 100) / 100.0;

    // Allocazione iniziale dei parametri su host
    float h_means[k * d] = {0.5, 0.5, 0.3, 0.7};
    float h_covMatrices[k * d * d] = {1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0};
    float h_weights[k] = {0.5, 0.5};

    // Allocazione GPU
    float *d_data, *d_means, *d_covMatrices, *d_weights, *d_responsibilities, *d_invCovMatrices;
    CUDA_CHECK(cudaMalloc(&d_data, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, k * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covMatrices, k * d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_invCovMatrices, k * d * d * sizeof(float)));


    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, h_means, k * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covMatrices, h_covMatrices, k * d * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, k * sizeof(float), cudaMemcpyHostToDevice));

    // Handle cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    for (int iter = 0; iter < maxIter; ++iter) {
    printf("Iterazione %d\n", iter + 1);

    // E-step: calcolo delle inverse delle matrici di covarianza
    computeInverseMatrices(handle, d_covMatrices, d, k, d_invCovMatrices);
    cudaDeviceSynchronize();

    // E-step: calcolo responsabilità
    computeResponsibilities<<<(N + 255) / 256, 256>>>(
        d_data, d_means, d_invCovMatrices, d_weights, d_weights,
        d_responsibilities, d, d, k, N);
    cudaDeviceSynchronize();

    normalizeResponsibilities<<<(N + 255) / 256, 256>>>(d_responsibilities, k, N);
    cudaDeviceSynchronize();

    // M-step: aggiornamento parametri
    mStep<<<(k + 255) / 256, 256>>>(
        d_data, d_responsibilities, d_means, d_covMatrices,
        d_weights, d, d, k, N);
    cudaDeviceSynchronize();
}

    // Copia dei risultati su host
    CUDA_CHECK(cudaMemcpy(h_means, d_means, k * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_covMatrices, d_covMatrices, k * d * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Risultati finali:\n");
    for (int i = 0; i < k; ++i) {
        printf("Cluster %d:\n", i + 1);
        printf("Mean: ");
        for (int j = 0; j < d; ++j) {
            printf("%f ", h_means[i * d + j]);
        }
        printf("\nCovariance Matrix:\n");
        for (int j = 0; j < d; ++j) {
            for (int l = 0; l < d; ++l) {
                printf("%f ", h_covMatrices[i * d * d + j * d + l]); // vet[i][j][l]
            }
            printf("\n");
        }
        printf("Weight: %f\n", h_weights[i]);
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_covMatrices);
    cudaFree(d_weights);
    cudaFree(d_responsibilities);
    cublasDestroy(handle);

    return 0;
}  */

/* int main() {
    const int d = 2;    // Dimensione delle feature
    const int k = 2;      // Numero di cluster
    const int N = 10000; // Numero di campioni (1 milione)
    const int maxIter = 100000;

    // Generazione dati sintetici
    float* h_data = (float*)malloc(N * d * sizeof(float));

    for (int i = 0; i < (1/3) * (N * d); ++i) h_data[i] = (float)(rand() % 100) / 100.0;

     for (int i = (1/3) * (N * d); i < N * d; ++i) h_data[i] = -1 * ((float)(rand() % 100) / 100.0);

    // Allocazione iniziale dei parametri su host
    float* h_means = (float*)malloc(k * d * sizeof(float));
    float* h_covMatrices = (float*)malloc(k * d * d * sizeof(float));
    float h_weights[k] = {0.2, 0.8, 0.34 };

    for (int i = 0; i < k * d; ++i) {
        h_means[i] = (float)(rand() % 100) / 100.0;
    }

    for(int i=0; i< k; ++i) {
        for(int j=0; j<d; ++j) {
            for (int l=0; l<d; ++l) {
                if(j==l) {
                    h_covMatrices[i * d * d + j * d + l] = 1.0;
                } else {
                    h_covMatrices[i * d * d + j * d + l] = 0.0;
                }
                printf("%f ", h_covMatrices[i * d * d + j * d + l]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Allocazione GPU
    float *d_data, *d_means, *d_covMatrices, *d_weights, *d_responsibilities, *d_invCovMatrices;
    CUDA_CHECK(cudaMalloc(&d_data, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, k * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covMatrices, k * d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_invCovMatrices, k * d * d * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, h_means, k * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covMatrices, h_covMatrices, k * d * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, k * sizeof(float), cudaMemcpyHostToDevice));

    // Handle cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Eventi CUDA per temporizzazione
    cudaEvent_t start, stop, iterStart, iterStop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&iterStart));
    CUDA_CHECK(cudaEventCreate(&iterStop));

    // Avvio temporizzazione totale
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < maxIter; ++iter) {
        printf("Iterazione %d\n", iter + 1);

        // Inizio temporizzazione iterazione
        CUDA_CHECK(cudaEventRecord(iterStart));

        // E-step: calcolo delle inverse delle matrici di covarianza
        computeInverseMatrices(handle, d_covMatrices, d, k, d_invCovMatrices);
        cudaDeviceSynchronize();

        computeResponsibilities<<<(N + 255) / 256, 256>>>(
            d_data, d_means, d_invCovMatrices, d_weights, d_weights,
            d_responsibilities, d, d, k, N);
        cudaDeviceSynchronize();

        normalizeResponsibilities<<<(N + 255) / 256, 256>>>(d_responsibilities, k, N);
        cudaDeviceSynchronize();

        // M-step: aggiornamento parametri
        mStep<<<(k + 255) / 256, 256>>>(
            d_data, d_responsibilities, d_means, d_covMatrices,
            d_weights, d, d, k, N);
        cudaDeviceSynchronize();

        // Fine temporizzazione iterazione
        CUDA_CHECK(cudaEventRecord(iterStop));
        CUDA_CHECK(cudaEventSynchronize(iterStop));

        float iterTime = 0;
        CUDA_CHECK(cudaEventElapsedTime(&iterTime, iterStart, iterStop));
        printf("Iterazione %d time: %f ms\n", iter + 1, iterTime);
    }

    // Fine temporizzazione totale
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    printf("Total time: %f ms\n", totalTime);

    // Copia dei risultati su host
    CUDA_CHECK(cudaMemcpy(h_means, d_means, k * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_covMatrices, d_covMatrices, k * d * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Risultati finali:\n");
    for (int i = 0; i < k; ++i) {
        printf("Cluster %d:\n", i + 1);
        printf("Mean: ");
        for (int j = 0; j < d; ++j) {
            printf("%f ", h_means[i * d + j]);
        }
        printf("\nCovariance Matrix:\n");
        for (int j = 0; j < d; ++j) {
            for (int l = 0; l < d; ++l) {
                printf("%f ", h_covMatrices[i * d * d + j * d + l]); // vet[i][j][l]
            }
            printf("\n");
        }
        printf("Weight: %f\n", h_weights[i]);
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_covMatrices);
    cudaFree(d_weights);
    cudaFree(d_responsibilities);
    free(h_data);
    free(h_means);
    free(h_covMatrices);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);

    return 0;
}
 */

 #include <stdio.h>
#include <stdlib.h>

int main() {
    const int d = 2;    // Dimensione delle feature
    const int k = 2;    // Numero di cluster
    const int maxIter = 1000;
    const char* fileName = "data.csv"; // Nome del file CSV

    // Lettura dei dati dal CSV
    FILE* file = fopen(fileName, "r");
    if (file == NULL) {
        perror("Errore nell'apertura del file CSV");
        return EXIT_FAILURE;
    }

    // Contiamo il numero di righe per determinare N
    int N = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)) {
        N++;
    }

    // Allochiamo memoria per i dati
    float* h_data = (float*)malloc(N * d * sizeof(float));
    if (h_data == NULL) {
        perror("Errore nell'allocazione della memoria per i dati");
        fclose(file);
        return EXIT_FAILURE;
    }

    // Torniamo all'inizio del file e leggiamo i valori
    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        if (sscanf(line, "%f,%f", &h_data[i * d], &h_data[i * d + 1]) != 2) {
            fprintf(stderr, "Errore nella lettura del file CSV alla riga %d\n", i + 1);
            free(h_data);
            fclose(file);
            return EXIT_FAILURE;
        }
        i++;
    }
    fclose(file);

    // Allocazione iniziale dei parametri su host
    float* h_means = (float*)malloc(k * d * sizeof(float));
    float* h_covMatrices = (float*)malloc(k * d * d * sizeof(float));
    float h_weights[k] = {0.2, 0.8};

    for (int i = 0; i < k * d; ++i) {
        h_means[i] = (float)(rand() % 100) / 100.0;
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            for (int l = 0; l < d; ++l) {
                if (j == l) {
                    h_covMatrices[i * d * d + j * d + l] = 1.0;
                } else {
                    h_covMatrices[i * d * d + j * d + l] = 0.0;
                }
            }
        }
    }

    // Allocazione GPU
    float *d_data, *d_means, *d_covMatrices, *d_weights, *d_responsibilities, *d_invCovMatrices;
    CUDA_CHECK(cudaMalloc(&d_data, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, k * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covMatrices, k * d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_invCovMatrices, k * d * d * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, h_means, k * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covMatrices, h_covMatrices, k * d * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, k * sizeof(float), cudaMemcpyHostToDevice));

    // Handle cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Eventi CUDA per temporizzazione
    cudaEvent_t start, stop, iterStart, iterStop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventCreate(&iterStart));
    CUDA_CHECK(cudaEventCreate(&iterStop));

    // Avvio temporizzazione totale
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < maxIter; ++iter) {
        printf("Iterazione %d\n", iter + 1);

        // Inizio temporizzazione iterazione
        CUDA_CHECK(cudaEventRecord(iterStart));

        // E-step: calcolo delle inverse delle matrici di covarianza
        computeInverseMatrices(handle, d_covMatrices, d, k, d_invCovMatrices);
        cudaDeviceSynchronize();

        computeResponsibilities<<<(N + 255) / 256, 256>>>(
            d_data, d_means, d_invCovMatrices, d_weights, d_weights,
            d_responsibilities, d, d, k, N);
        cudaDeviceSynchronize();

        normalizeResponsibilities<<<(N + 255) / 256, 256>>>(d_responsibilities, k, N);
        cudaDeviceSynchronize();

        // M-step: aggiornamento parametri
        mStep<<<(k + 255) / 256, 256>>>(
            d_data, d_responsibilities, d_means, d_covMatrices,
            d_weights, d, d, k, N);
        cudaDeviceSynchronize();

        // Fine temporizzazione iterazione
        CUDA_CHECK(cudaEventRecord(iterStop));
        CUDA_CHECK(cudaEventSynchronize(iterStop));

        float iterTime = 0;
        CUDA_CHECK(cudaEventElapsedTime(&iterTime, iterStart, iterStop));
        printf("Iterazione %d time: %f ms\n", iter + 1, iterTime);
    }

    // Fine temporizzazione totale
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    printf("Total time: %f s\n", totalTime / 1000.0);

    // Copia dei risultati su host
    CUDA_CHECK(cudaMemcpy(h_means, d_means, k * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_covMatrices, d_covMatrices, k * d * d * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Risultati finali:\n");
    for (int i = 0; i < k; ++i) {
        printf("Cluster %d:\n", i + 1);
        printf("Mean: ");
        for (int j = 0; j < d; ++j) {
            printf("%f ", h_means[i * d + j]);
        }
        printf("\nCovariance Matrix:\n");
        for (int j = 0; j < d; ++j) {
            for (int l = 0; l < d; ++l) {
                printf("%f ", h_covMatrices[i * d * d + j * d + l]); // vet[i][j][l]
            }
            printf("\n");
        }
        printf("Weight: %f\n", h_weights[i]);
    }

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_covMatrices);
    cudaFree(d_weights);
    cudaFree(d_responsibilities);
    free(h_data);
    free(h_means);
    free(h_covMatrices);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(iterStart);
    cudaEventDestroy(iterStop);

    return 0;
}


