#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <float.h>

#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
    exit(EXIT_FAILURE); \
}

#define CUBLAS_CHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at line %d\n", __LINE__); \
    exit(EXIT_FAILURE); \
}

__global__ void computeResponsibilities(
    const float* data, const float* means, const float* invCovMatrices,
    const float* determinants, const float* weights,
    float* responsibilities, int d, int k, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx >= N) return;
    

    for (int i = idx; i < N; i += gridDim.x * blockDim.x) {
        float sum = 0.0f; 
        float diff[32];
        float temp[32];

        for (int cluster = 0; cluster < k; ++cluster) {
            // Calcola la differenza data - mean per il cluster
            for (int j = 0; j < d; ++j) {
                diff[j] = data[i * d + j] - means[cluster * d + j];
            }

            // Calcola il prodotto invCovMatrix * diff
            for (int j = 0; j < d; ++j) {
                temp[j] = 0.0f;
                for (int l = 0; l < d; ++l) {
                    temp[j] += invCovMatrices[cluster * d * d + j * d + l] * diff[l];
                }
            }

            // Calcola la distanza di Mahalanobis
            float mahalanobis = 0.0f;
            for (int j = 0; j < d; ++j) {
                mahalanobis += diff[j] * temp[j];
            }

            // Calcola la verosimiglianza
            float likelihood = expf(-0.5f * mahalanobis) / 
                                sqrtf(powf(2 * M_PI, d) * determinants[cluster]);

            // Calcola la responsabilità pesata
            responsibilities[i * k + cluster] = weights[cluster] * likelihood;
            sum += responsibilities[i * k + cluster];
        }

        // Normalizzazione delle responsabilità
        for (int cluster = 0; cluster < k; ++cluster) {
            // if sum is near 0, set the responsibility to 0
            if (sum == 0) {
                responsibilities[i * k + cluster] = 0.0;
            } else {
                responsibilities[i * k + cluster] /= sum;
            }
        } 
    }
}

__global__ void mStep(
    const float* data, const float* responsibilities, float* means,
    float* covMatrices, float* weights, int d, int k, int N) {

    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) return;

    float weightSum = 0.0;

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

    for (int i = 0; i < d * d; ++i) {
        covMatrices[cluster * d * d + i] = 0.0; // [cluster][d][i]
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

   for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            covMatrices[cluster * d * d + i * d + j] /= weightSum;
            // Aggiungi il termine di regolarizzazione alla diagonale
            if (i == j) {
                covMatrices[cluster * d * d + i * d + j] += 0.0001;
            }
        }
    }

    weights[cluster] = weightSum / N;
}

void computeInverseMatrices(
    cublasHandle_t handle, float* d_matrices, int d, int batchSize,
    float* d_invMatrices, float* d_determinants) {

    float** d_matrixArray;
    CUDA_CHECK(cudaMalloc((void**)&d_matrixArray, batchSize * sizeof(float*)));
    float** d_invMatrixArray;
    CUDA_CHECK(cudaMalloc((void**)&d_invMatrixArray, batchSize * sizeof(float*)));

    for (int i = 0; i < batchSize; ++i) {
        float* matrixAddress = d_matrices + i * d * d;
        float* invMatrixAddress = d_invMatrices + i * d * d;

        CUDA_CHECK(cudaMemcpy(d_matrixArray + i, &matrixAddress, sizeof(float*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_invMatrixArray + i, &invMatrixAddress, sizeof(float*), cudaMemcpyHostToDevice));
    }

    int* d_pivotArray;
    int* d_infoArray;
    CUDA_CHECK(cudaMalloc((void**)&d_pivotArray, batchSize * d * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_infoArray, batchSize * sizeof(int)));

    CUBLAS_CHECK(cublasSgetrfBatched(handle, d, d_matrixArray, d, d_pivotArray, d_infoArray, batchSize));
    
    int * h_pivotArray = (int*)malloc(batchSize * d * sizeof(int));
    float * h_matrixArray = (float*)malloc(batchSize * d * d * sizeof(float));
    float * h_determinants = (float*)malloc(batchSize * sizeof(float));

    cudaMemcpy(h_pivotArray, d_pivotArray, batchSize * d * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_matrixArray, d_matrices, batchSize * d * d * sizeof(float), cudaMemcpyDeviceToHost);
    

    for (int i = 0; i < batchSize; i++) {
        float det = 1.0f;  // Inizializza a 1.0 per il prodotto
        int swaps = 0;

        for (int j = 0; j < d; j++) {
            // Moltiplicazione di tutti gli elementi diagonali
            det *= h_matrixArray[i * d * d + j * d + j]; // [i][j][j]

            // Controlla se il pivot non è nella posizione attesa
            if (h_pivotArray[i * d + j] != j + 1) { // [i][j]
                swaps++;
            }
        }

        // Cambia segno al determinante se il numero di scambi è dispari
        if (swaps % 2 != 0) {
            det = -det;
        }

        h_determinants[i] = det;
        // printf("Determinante %d: %f\n", i + 1, h_determinants[i]);
    }

    // Copia i determinanti sul dispositivo
    cudaMemcpy(d_determinants, h_determinants, batchSize * sizeof(float), cudaMemcpyHostToDevice);

    CUBLAS_CHECK(cublasSgetriBatched(handle, d, (const float**)d_matrixArray, d, d_pivotArray, d_invMatrixArray, d, d_infoArray, batchSize));

   /*  
    printf("Matrici inverse:\n");
    float * h_invMatrixArray = (float*)malloc(batchSize * d * d * sizeof(float));
    cudaMemcpy(h_invMatrixArray, d_invMatrices, batchSize * d * d * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize; i++) {
        printf("Matrice %d:\n", i + 1);
        for (int j = 0; j < d; j++) {
            for (int k = 0; k < d; k++) {
                printf("%f ", h_invMatrixArray[i * d * d + j * d + k]);
            }
            printf("\n");
        }
    } */


    free(h_pivotArray);
    free(h_matrixArray);
    free(h_determinants);
    cudaFree(d_matrixArray);
    cudaFree(d_invMatrixArray);
    cudaFree(d_pivotArray);
    cudaFree(d_infoArray);
}

int main() {
    const int d = 10;    // Cambia il valore per il numero di feature desiderato
    const int k = 5;    // Numero di cluster
    const int maxIter = 2;
    const char* fileName = "data.csv"; // Nome del file CSV
    int threadsPerBlock = 256;
    int dataPerThread = 100;


    FILE* file = fopen(fileName, "r");
    if (file == NULL) {
        perror("Errore nell'apertura del file CSV");
        return EXIT_FAILURE;
    }

    int N = 0;
    char line[1024];
    while (fgets(line, sizeof(line), file)) {
        N++;
    }

    float* h_data = (float*)malloc(N * d * sizeof(float));
    if (h_data == NULL) {
        perror("Errore nell'allocazione della memoria per i dati");
        fclose(file);
        return EXIT_FAILURE;
    }

    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, ",");
        for (int j = 0; j < d; ++j) {
            if (token != NULL) {
                h_data[i * d + j] = atof(token);
                token = strtok(NULL, ",");
            } else {
                fprintf(stderr, "Errore nella lettura del file CSV alla riga %d\n", i + 1);
                free(h_data);
                fclose(file);
                return EXIT_FAILURE;
            }
        }
        i++;
    }
    fclose(file);

     // Calcolo dei valori minimi e massimi per ogni feature
/*     float min_values[d];
    float max_values[d];
    for (int j = 0; j < d; ++j) {
        min_values[j] = FLT_MAX;  // Impostiamo il valore minimo iniziale al massimo possibile
        max_values[j] = -FLT_MAX; // Impostiamo il valore massimo iniziale al minimo possibile
    }

    // Determiniamo i valori minimi e massimi per ciascuna feature
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            float value = h_data[i * d + j];
            if (value < min_values[j]) {
                min_values[j] = value;
            }
            if (value > max_values[j]) {
                max_values[j] = value;
            }
        }
    } */

  /*   for(int i = 0; i < d; i++) {
        printf("Feature %d: Min = %f, Max = %f\n", i + 1, min_values[i], max_values[i]);
    }
    printf("\n"); */

/*     // Normalizzazione Min-Max dei dati
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            float min_value = min_values[j];
            float max_value = max_values[j];
            if (max_value - min_value != 0) {
                // Normalizziamo il valore nell'intervallo [0, 1]
                h_data[i * d + j] = (h_data[i * d + j] - min_value) / (max_value - min_value);
            } else {
                // Se max_value == min_value, normalizzare non ha senso (la feature è costante), quindi impostiamo a 0.5 come valore di default
                h_data[i * d + j] = 0.5;
            }
        }
    } */

    float* h_means = (float*)malloc(k * d * sizeof(float));
    float* h_covMatrices = (float*)malloc(k * d * d * sizeof(float));
    float* h_weights = (float*)malloc(k * sizeof(float));

   /*  int maxBlocks;
    cudaDeviceGetAttribute(&maxBlocks, cudaDevAttrMultiProcessorCount, 0); */
   
    /* for (int i = 0; i < k * d; ++i) {
        h_means[i] = (float)(rand() % 100) / 100.0;
    } */
    float feature_means[d];
    for (int j = 0; j < d; ++j) {
        feature_means[j] = 0.0;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            feature_means[j] += h_data[i * d + j];
        }
    }

    for (int j = 0; j < d; ++j) {
        feature_means[j] /= N;
    }

    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < d; ++j) {
            h_means[i * d + j] = feature_means[j] + (float)(rand() % 100) / 100.0; // 5 + (0, 1)
        }
    }


    for (int i = 0; i < k; ++i) {
        h_weights[i] = 1.0 / k;
        for (int j = 0; j < d; ++j) {
            for (int l = 0; l < d; ++l) {
                h_covMatrices[i * d * d + j * d + l] = (j == l) ? 1.0 : 0.0;
            }
        }
    }

    float *d_data, *d_means, *d_covMatrices, *d_weights, *d_responsibilities, *d_invCovMatrices, *d_determinants;
    CUDA_CHECK(cudaMalloc(&d_data, N * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_means, k * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_covMatrices, k * d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * k * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_invCovMatrices, k * d * d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_determinants, k * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, h_means, k * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covMatrices, h_covMatrices, k * d * d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, k * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // Avvio temporizzazione totale
    CUDA_CHECK(cudaEventRecord(start));


    int totalThreads = (N + dataPerThread - 1) / dataPerThread;
    //  int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;
    int numBlocks = (N + threadsPerBlock * dataPerThread - 1) / (threadsPerBlock * dataPerThread);
    printf("Numero di blocchi: %d\n", numBlocks);
    printf("Numero di thread per blocco: %d\n", threadsPerBlock);
    printf("Numero totale di thread: %d\n", totalThreads);
    printf("Numero di dati per thread: %d\n", dataPerThread);

    int numSMs;
    cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
    printf("Numero di SMs: %d\n", numSMs);

    for (int iter = 0; iter < maxIter; ++iter) {
        // printf("Iterazione %d\n", iter + 1);

        computeInverseMatrices(handle, d_covMatrices, d, k, d_invCovMatrices, d_determinants);
        cudaDeviceSynchronize();
 
        computeResponsibilities<<<numBlocks, threadsPerBlock/* (N + 255) / 256, 256 */>>>(
            d_data, d_means, d_invCovMatrices, d_determinants, d_weights,
            d_responsibilities, d, k, N);
        cudaDeviceSynchronize();

        mStep<<<(k + 255) / 256, 256>>>(
            d_data, d_responsibilities, d_means, d_covMatrices,
            d_weights, d, k, N);
        cudaDeviceSynchronize();

/* 
        float* h_responsibilities = (float*)malloc(N * k * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_responsibilities, d_responsibilities, N * k * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_means, d_means, k * d * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_covMatrices, d_covMatrices, k * d * d * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(float), cudaMemcpyDeviceToHost));
        // print first responsibility, first mean, first covariance matrix and first weight
         printf("\n");
          printf("\n");

        printf("Data 1: ");
        for (int j = 0; j < d; ++j) {
            printf("%f ", h_data[j]);
        }
        printf("\n");
        printf("Responsibility 1: ");
        for (int j = 0; j < k; ++j) {
            printf("%f ", h_responsibilities[j]);
        }
        printf("\n");
        printf("Mean 1: ");
        for (int j = 0; j < d; ++j) {
            printf("%f ", h_means[j]);
        }
        printf("\n");
        printf("Covariance Matrix 1:\n");
        for (int j = 0; j < d; ++j) {
            for (int l = 0; l < d; ++l) {
                printf("%f ", h_covMatrices[j * d + l]);
            }
            printf("\n");
        }
        printf("Inv Covariance Matrix 1:\n");
        float* h_invCovMatrices = (float*)malloc(k * d * d * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_invCovMatrices, d_invCovMatrices, k * d * d * sizeof(float), cudaMemcpyDeviceToHost));
        for (int j = 0; j < d; ++j) {
            for (int l = 0; l < d; ++l) {
                printf("%f ", h_invCovMatrices[j * d + l]);
            }
            printf("\n");
        }

        
        printf("Weight 1: %f\n", h_weights[0]);
        free(h_responsibilities);
        printf("\n");
        printf("\n");
         */

    }

    // Fine temporizzazione totale
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    printf("Total time: %f s\n\n", totalTime / 1000.0);

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
                printf("%f ", h_covMatrices[i * d * d + j * d + l]);
            }
            printf("\n");
        }
        printf("Weight: %f\n", h_weights[i]);
    }

    cudaFree(d_data);
    cudaFree(d_means);
    cudaFree(d_covMatrices);
    cudaFree(d_weights);
    cudaFree(d_responsibilities);
    free(h_data);
    free(h_means);
    free(h_covMatrices);
    free(h_weights);
    cublasDestroy(handle);

    return 0;
}