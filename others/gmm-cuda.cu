#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err) if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
    exit(EXIT_FAILURE); \
}

#define CUBLAS_CHECK(err) if (err != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at line %d\n", __LINE__); \
    exit(EXIT_FAILURE); \
}

__global__ void computeResponsibilities(
    const double* data, const double* means, const double* invCovMatrices,
    const double* determinants, const double* weights,
    double* responsibilities, int d, int k, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    double sum = 0.0; 
    double diff[32];
    double temp[32];

    for (int cluster = 0; cluster < k; ++cluster) {
        for (int i = 0; i < d; ++i) {
            diff[i] = data[idx * d + i] - means[cluster * d + i];
        }
        // Calcola il prodotto tra la matrice inversa e il vettore differenza
        for (int i = 0; i < d; ++i) {
            temp[i] = 0.0;
            for (int j = 0; j < d; ++j) {
                temp[i] += invCovMatrices[cluster * d * d + i * d + j] * diff[j];
            }
        }
        // Calcola il prodotto tra il vettore differenza e il vettore risultante
        double mahalanobis = 0.0;
        for (int i = 0; i < d; ++i) {
            mahalanobis += diff[i] * temp[i];
        }

        double likelihood = exp(-0.5 * mahalanobis) / sqrt(pow(2 * M_PI, d) * determinants[cluster]);
        responsibilities[idx * k + cluster] = weights[cluster] * likelihood;
        sum += responsibilities[idx * k + cluster];
    }


   for (int cluster = 0; cluster < k; ++cluster) {
        // if sum is near 0, set the responsibility to 1/k
        if (sum == 0) {
            responsibilities[idx * k + cluster] = 1.0 / k;
        } else {
            responsibilities[idx * k + cluster] /= sum;
        }
    } 
}

__global__ void mStep(
    const double* data, const double* responsibilities, double* means,
    double* covMatrices, double* weights, int d, int k, int N) {

    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) return;

    double weightSum = 0.0;

    for (int i = 0; i < d; ++i) {
        means[cluster * d + i] = 0.0;
    }

    for (int idx = 0; idx < N; ++idx) {
        double r = responsibilities[idx * k + cluster]; 
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
        double r = responsibilities[idx * k + cluster];
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                double diff_i = data[idx * d + i] - means[cluster * d + i];
                double diff_j = data[idx * d + j] - means[cluster * d + j];
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
    cublasHandle_t handle, double* d_matrices, int d, int batchSize,
    double* d_invMatrices, double* d_determinants) {

    double** d_matrixArray;
    CUDA_CHECK(cudaMalloc((void**)&d_matrixArray, batchSize * sizeof(double*)));
    double** d_invMatrixArray;
    CUDA_CHECK(cudaMalloc((void**)&d_invMatrixArray, batchSize * sizeof(double*)));

    for (int i = 0; i < batchSize; ++i) {
        double* matrixAddress = d_matrices + i * d * d;
        double* invMatrixAddress = d_invMatrices + i * d * d;

        CUDA_CHECK(cudaMemcpy(d_matrixArray + i, &matrixAddress, sizeof(double*), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_invMatrixArray + i, &invMatrixAddress, sizeof(double*), cudaMemcpyHostToDevice));
    }

    int* d_pivotArray;
    int* d_infoArray;
    CUDA_CHECK(cudaMalloc((void**)&d_pivotArray, batchSize * d * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_infoArray, batchSize * sizeof(int)));

    CUBLAS_CHECK(cublasDgetrfBatched(handle, d, d_matrixArray, d, d_pivotArray, d_infoArray, batchSize));
    
    int * h_pivotArray = (int*)malloc(batchSize * d * sizeof(int));
    double * h_matrixArray = (double*)malloc(batchSize * d * d * sizeof(double));
    double * h_determinants = (double*)malloc(batchSize * sizeof(double));

    cudaMemcpy(h_pivotArray, d_pivotArray, batchSize * d * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_matrixArray, d_matrices, batchSize * d * d * sizeof(double), cudaMemcpyDeviceToHost);
    

    for (int i = 0; i < batchSize; i++) {
        double det = 1.0f;  // Inizializza a 1.0 per il prodotto
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
    cudaMemcpy(d_determinants, h_determinants, batchSize * sizeof(double), cudaMemcpyHostToDevice);

    CUBLAS_CHECK(cublasDgetriBatched(handle, d, (const double**)d_matrixArray, d, d_pivotArray, d_invMatrixArray, d, d_infoArray, batchSize));

   /*  
    printf("Matrici inverse:\n");
    double * h_invMatrixArray = (double*)malloc(batchSize * d * d * sizeof(double));
    cudaMemcpy(h_invMatrixArray, d_invMatrices, batchSize * d * d * sizeof(double), cudaMemcpyDeviceToHost);

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
    const int maxIter = 100;
    const char* fileName = "data.csv"; // Nome del file CSV

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

    double* h_data = (double*)malloc(N * d * sizeof(double));
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
/*     double min_values[d];
    double max_values[d];
    for (int j = 0; j < d; ++j) {
        min_values[j] = FLT_MAX;  // Impostiamo il valore minimo iniziale al massimo possibile
        max_values[j] = -FLT_MAX; // Impostiamo il valore massimo iniziale al minimo possibile
    }

    // Determiniamo i valori minimi e massimi per ciascuna feature
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < d; ++j) {
            double value = h_data[i * d + j];
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
            double min_value = min_values[j];
            double max_value = max_values[j];
            if (max_value - min_value != 0) {
                // Normalizziamo il valore nell'intervallo [0, 1]
                h_data[i * d + j] = (h_data[i * d + j] - min_value) / (max_value - min_value);
            } else {
                // Se max_value == min_value, normalizzare non ha senso (la feature è costante), quindi impostiamo a 0.5 come valore di default
                h_data[i * d + j] = 0.5;
            }
        }
    } */

    double* h_means = (double*)malloc(k * d * sizeof(double));
    double* h_covMatrices = (double*)malloc(k * d * d * sizeof(double));
    double* h_weights = (double*)malloc(k * sizeof(double));

    /* for (int i = 0; i < k * d; ++i) {
        h_means[i] = (double)(rand() % 100) / 100.0;
    } */
    double feature_means[d];
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
            h_means[i * d + j] = feature_means[j] + (double)(rand() % 100) / 100.0; // 5 + (0, 1)
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

    double *d_data, *d_means, *d_covMatrices, *d_weights, *d_responsibilities, *d_invCovMatrices, *d_determinants;
    CUDA_CHECK(cudaMalloc(&d_data, N * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_means, k * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_covMatrices, k * d * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weights, k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_invCovMatrices, k * d * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_determinants, k * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * d * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_means, h_means, k * d * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_covMatrices, h_covMatrices, k * d * d * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, k * sizeof(double), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    // Avvio temporizzazione totale
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < maxIter; ++iter) {
        // printf("Iterazione %d\n", iter + 1);
        computeInverseMatrices(handle, d_covMatrices, d, k, d_invCovMatrices, d_determinants);
        cudaDeviceSynchronize();
 
        computeResponsibilities<<<(N + 255) / 256, 256>>>(
            d_data, d_means, d_invCovMatrices, d_determinants, d_weights,
            d_responsibilities, d, k, N);
        cudaDeviceSynchronize();

        mStep<<<(k + 255) / 256, 256>>>(
            d_data, d_responsibilities, d_means, d_covMatrices,
            d_weights, d, k, N);
        cudaDeviceSynchronize();

        /* 
        // print weights
        CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(double), cudaMemcpyDeviceToHost));
        printf("Weights:\n");
        for (int i = 0; i < k; ++i) {
            printf("%f ", h_weights[i]);
        }
        printf("\n");

        // print responsibilities
        double* h_responsibilities = (double*)malloc(N * k * sizeof(double));
        CUDA_CHECK(cudaMemcpy(h_responsibilities, d_responsibilities, N * k * sizeof(double), cudaMemcpyDeviceToHost));
        printf("Responsibilities:\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < k; ++j) {
                printf("%f ", h_responsibilities[i * k + j]);
            }
            printf("\n");
        }

        free(h_responsibilities);

        // print means
        printf("\n");
        CUDA_CHECK(cudaMemcpy(h_means, d_means, k * d * sizeof(double), cudaMemcpyDeviceToHost));
        printf("Means:\n");
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < d; ++j) {
                printf("%f ", h_means[i * d + j]);
            }
            printf("\n");
        } */


    }

    // Fine temporizzazione totale
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float totalTime = 0;
    CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    printf("Total time: %f s\n\n", totalTime / 1000.0);

    CUDA_CHECK(cudaMemcpy(h_means, d_means, k * d * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_covMatrices, d_covMatrices, k * d * d * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(double), cudaMemcpyDeviceToHost));

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
