#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(err)                                                                    \
    if (err != cudaSuccess)                                                                \
    {                                                                                      \
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE);                                                                \
    }

#define CUBLAS_CHECK(err)                                       \
    if (err != CUBLAS_STATUS_SUCCESS)                           \
    {                                                           \
        fprintf(stderr, "cuBLAS error at line %d\n", __LINE__); \
        exit(EXIT_FAILURE);                                     \
    }

__global__ void computeResponsibilities(
    const double *data, const double *means, const double *invCovMatrices,
    const double *determinants, const double *weights,
    double *responsibilities, double *local_means, double *local_weights, int d, int k, int N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (idx >= N) return;

    // reset local means and weights
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            local_means[idx * k * d + i * d + j] = 0.0; //[idx][i][j]
        }
        local_weights[idx * k + i] = 0.0; //[idx][i]
    }

    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
    {
        double sum = 0.0;
        double diff[32];
        double temp[32];

        for (int cluster = 0; cluster < k; ++cluster)
        {
            // Calcola la differenza data - mean per il cluster
            for (int j = 0; j < d; ++j)
            {
                diff[j] = data[i * d + j] - means[cluster * d + j];
            }

            // Calcola il prodotto invCovMatrix * diff
            for (int j = 0; j < d; ++j)
            {
                temp[j] = 0.0;
                for (int l = 0; l < d; ++l)
                {
                    temp[j] += invCovMatrices[cluster * d * d + j * d + l] * diff[l];
                }
            }

            // Calcola la distanza di Mahalanobis
            double mahalanobis = 0.0;
            for (int j = 0; j < d; ++j)
            {
                mahalanobis += diff[j] * temp[j];
            }

            // Calcola la verosimiglianza
            double likelihood = exp(-0.5 * mahalanobis) /
                                sqrt(pow(2 * M_PI, d) * determinants[cluster]);

            // Calcola la responsabilità pesata
            responsibilities[i * k + cluster] = weights[cluster] * likelihood;
            sum += responsibilities[i * k + cluster];
        }

        // Normalizzazione delle responsabilità
        for (int cluster = 0; cluster < k; ++cluster)
        {
            // if sum is near 0, set the responsibility to 1/k
            if (sum == 0)
            {
                responsibilities[i * k + cluster] = 1.0 / k;
            }
            else
            {
                responsibilities[i * k + cluster] /= sum;
            }
            local_weights[idx * k + cluster] += responsibilities[i * k + cluster]; //[idx][cluster]

            for (int j = 0; j < d; ++j)
            {
                local_means[idx * k * d + cluster * d + j] += responsibilities[i * k + cluster] * data[i * d + j]; //[idx][cluster][j]
            }
        }
    }
}

__global__ void reduceWeightMean(
    double *local_means, double *local_weights, double *means, double *weights, int d, int k, int Num, int elemsPerThread)
{
    // create shared memory for local means and weights
    __shared__ double s_means[256 * 10];
    __shared__ double s_weights[256];

    int cluster = blockIdx.x;
    unsigned int startIdx = threadIdx.x * elemsPerThread;

    double sumWeights = 0.0;
    double sumMeans[32] = {0.0};
    // stampare grandezza local_means per verificare un illegal memory access

    // [1 , 2, 3, 4]
    for (unsigned int offset = 0; offset < elemsPerThread; ++offset)
    {
        int idx = startIdx + offset;
        // stampare idx e N per verificare un illegal memory access
        if ((idx * (cluster + 1)) < Num)
        {
            sumWeights += local_weights[idx * k + cluster]; // [idx][cluster]
            for (int i = 0; i < d; ++i)
            {
                sumMeans[i] += local_means[idx * k * d + cluster * d + i];
            }
        }
    }

    s_weights[threadIdx.x] = sumWeights;

    for (int i = 0; i < d; ++i)
    {
        s_means[threadIdx.x * d + i] = sumMeans[i];
    }
    __syncthreads();

    // reduce local means and weights
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {
            s_weights[threadIdx.x] += s_weights[threadIdx.x + s];
            for (int i = 0; i < d; ++i)
            {
                s_means[threadIdx.x * d + i] += s_means[(threadIdx.x + s) * d + i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        weights[cluster] = s_weights[0] /* / N */;
        for (int i = 0; i < d; ++i)
        {
            means[cluster * d + i] = s_means[i] / s_weights[0];
        }

        // print means
    }
}

__global__ void reduceCovMatrices(
    double *local_cov_matrices, double *cov_matrices, double *weights, int d, int k, int num, int N, int elemsPerThread)
{
    // create shared memory for local covariance matrices
    extern __shared__ double s_cov_matrices[];
    __shared__ double s_weights[256];

    int cluster = blockIdx.x;
    unsigned int startIdx = threadIdx.x * elemsPerThread;

    double sumWeights = 0.0;
    double sumCovMatrices[10 * 10] = {0.0}; // assuming d <= 32

    for (unsigned int offset = 0; offset < elemsPerThread; ++offset)
    {
        int idx = startIdx + offset;
        if ((idx * (cluster + 1)) < num)
        {

            for (int j = 0; j < d; ++j)
            {
                for (int l = 0; l < d; ++l)
                {
                    sumCovMatrices[j * d + l] += local_cov_matrices[idx * k * d * d + cluster * d * d + j * d + l];
                }
            }
        }
    }

    for (int j = 0; j < d; ++j)
    {
        for (int l = 0; l < d; ++l)
        {
            s_cov_matrices[threadIdx.x * d * d + j * d + l] = sumCovMatrices[j * d + l];
        }
    }
    __syncthreads();

    // reduce local covariance matrices and weights
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (threadIdx.x < s)
        {

            for (int j = 0; j < d; ++j)
            {
                for (int l = 0; l < d; ++l)
                {
                    s_cov_matrices[threadIdx.x * d * d + j * d + l] += s_cov_matrices[(threadIdx.x + s) * d * d + j * d + l];
                }
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        for (int j = 0; j < d; ++j)
        {
            for (int l = 0; l < d; ++l)
            {
                cov_matrices[cluster * d * d + j * d + l] = s_cov_matrices[j * d + l] / weights[cluster];
                if (j == l)
                {
                    cov_matrices[cluster * d * d + j * d + l] += 0.0001;
                }
            }
        }
        weights[cluster] /= N;
    }
}

__global__ void mStep(
    const double *data, const double *responsibilities, double *means,
    double *local_cov_matrixes, int d, int k, int N)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int cluster = 0; cluster < k; cluster++)
    {
        for (int j = 0; j < d; j++)
        {
            for (int l = 0; l < d; l++)
            {
                local_cov_matrixes[idx * k * d * d + cluster * d * d + j * d + l] = 0.0; //[idx][cluster][j][l]
            }
        }
    }

    for (int i = idx; i < N; i += gridDim.x * blockDim.x)
    {
        for (int cluster = 0; cluster < k; cluster++)
        {
            double r = responsibilities[i * k + cluster];
            for (int j = 0; j < d; j++)
            {
                for (int l = 0; l < d; l++)
                {
                    double diff_j = data[i * d + j] - means[cluster * d + j];
                    double diff_l = data[i * d + l] - means[cluster * d + l];
                    local_cov_matrixes[idx * k * d * d + cluster * d * d + j * d + l] += r * diff_j * diff_l; // [idx][cluster][j][l]
                    // printf("Local cov matrixes %d: %f\n", i * k * d * d + cluster * d * d + j * d + l, local_cov_matrixes[i * k * d * d + cluster * d * d + j * d + l]);
                }
            }
        }
    }

    // print local cov matrixes
    /*   printf("Local cov matrixes for thread %d:\n", idx);
      for (int cluster = 0; cluster < k; ++cluster) {
          for (int j = 0; j < d; ++j) {
              for (int l = 0; l < d; ++l) {
                  printf("%f ", local_cov_matrixes[idx * k * d * d + cluster * d * d + j * d + l]);
              }
              printf("\n");
          }
          printf("\n\n");
      } */
}

/* __global__ void mStep(
    const double* data, const double* responsibilities, double* means,
    double* covMatrices, double* weights, int d, int k, int N) {

    int cluster = blockIdx.x * blockDim.x + threadIdx.x;
    if (cluster >= k) return;

    double weightSum = weights[cluster];

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

    weights[cluster] = weights[cluster] / N;
} */

void computeInverseMatrices(
    cublasHandle_t handle, double *d_matrices, int d, int batchSize,
    double *d_invMatrices, double *d_determinants)
{

    double **d_matrixArray;
    CUDA_CHECK(cudaMalloc((void **)&d_matrixArray, batchSize * sizeof(double *)));
    double **d_invMatrixArray;
    CUDA_CHECK(cudaMalloc((void **)&d_invMatrixArray, batchSize * sizeof(double *)));

    for (int i = 0; i < batchSize; ++i)
    {
        double *matrixAddress = d_matrices + i * d * d;
        double *invMatrixAddress = d_invMatrices + i * d * d;

        CUDA_CHECK(cudaMemcpy(d_matrixArray + i, &matrixAddress, sizeof(double *), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_invMatrixArray + i, &invMatrixAddress, sizeof(double *), cudaMemcpyHostToDevice));
    }

    int *d_pivotArray;
    int *d_infoArray;
    CUDA_CHECK(cudaMalloc((void **)&d_pivotArray, batchSize * d * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_infoArray, batchSize * sizeof(int)));

    CUBLAS_CHECK(cublasDgetrfBatched(handle, d, d_matrixArray, d, d_pivotArray, d_infoArray, batchSize));

    int *h_pivotArray = (int *)malloc(batchSize * d * sizeof(int));
    double *h_matrixArray = (double *)malloc(batchSize * d * d * sizeof(double));
    double *h_determinants = (double *)malloc(batchSize * sizeof(double));

    cudaMemcpy(h_pivotArray, d_pivotArray, batchSize * d * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_matrixArray, d_matrices, batchSize * d * d * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < batchSize; i++)
    {
        double det = 1.0f; // Inizializza a 1.0 per il prodotto
        int swaps = 0;

        for (int j = 0; j < d; j++)
        {
            // Moltiplicazione di tutti gli elementi diagonali
            det *= h_matrixArray[i * d * d + j * d + j]; // [i][j][j]

            // Controlla se il pivot non è nella posizione attesa
            if (h_pivotArray[i * d + j] != j + 1)
            { // [i][j]
                swaps++;
            }
        }

        // Cambia segno al determinante se il numero di scambi è dispari
        if (swaps % 2 != 0)
        {
            det = -det;
        }

        h_determinants[i] = det;
        // printf("Determinante %d: %f\n", i + 1, h_determinants[i]);
    }

    // Copia i determinanti sul dispositivo
    cudaMemcpy(d_determinants, h_determinants, batchSize * sizeof(double), cudaMemcpyHostToDevice);

    CUBLAS_CHECK(cublasDgetriBatched(handle, d, (const double **)d_matrixArray, d, d_pivotArray, d_invMatrixArray, d, d_infoArray, batchSize));

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

void checkCudaError(const char *message)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA error in %s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Funzione per calcolare la distanza euclidea al quadrato
double squaredEuclideanDistance(const double *point1, const double *point2, int d)
{
    double distance = 0.0;
    for (int i = 0; i < d; ++i)
    {
        double diff = point1[i] - point2[i];
        distance += diff * diff;
    }
    return distance;
}

// Funzione per inizializzare le medie con k-means++
void initializeMeans(double *h_data, double *h_means, int N, int d, int k)
{
    int *chosenIndices = (int *)malloc(k * sizeof(int));
    double *distances = (double *)malloc(N * sizeof(double));

    // Scegli il primo centro casualmente
    chosenIndices[0] = rand() % N;
    for (int i = 0; i < d; ++i)
    {
        h_means[i] = h_data[chosenIndices[0] * d + i];
    }

    // Inizializza le distanze
    for (int i = 0; i < N; ++i)
    {
        distances[i] = squaredEuclideanDistance(h_data + i * d, h_means, d);
    }

    // Scegli gli altri centri
    for (int cluster = 1; cluster < k; ++cluster)
    {
        double totalDistance = 0.0;
        for (int i = 0; i < N; ++i)
        {
            totalDistance += distances[i];
        }

        // Seleziona il prossimo centro basato sulla probabilità
        double r = ((double)rand() / RAND_MAX) * totalDistance;
        double cumulativeDistance = 0.0;
        int chosenIndex = -1;
        for (int i = 0; i < N; ++i)
        {
            cumulativeDistance += distances[i];
            if (cumulativeDistance >= r)
            {
                chosenIndex = i;
                break;
            }
        }
        chosenIndices[cluster] = chosenIndex;
        for (int i = 0; i < d; ++i)
        {
            h_means[cluster * d + i] = h_data[chosenIndices[cluster] * d + i];
        }

        // Aggiorna le distanze
        for (int i = 0; i < N; ++i)
        {
            double distance = squaredEuclideanDistance(h_data + i * d, h_means + cluster * d, d);
            if (distance < distances[i])
            {
                distances[i] = distance;
            }
        }
    }

    free(chosenIndices);
    free(distances);
}

int main()
{
    const int d = 10; // Numero di features
    const int k = 5;  // Numero di cluster
    const int maxIter = 5;
    const char *fileName = "../data/1M.csv"; // Nome del file CSV
    int threadsPerBlock = 256;
    int dataPerThread = 250;
    const double epsilon = 1e-9;
    double maxChange = 0.0;

    FILE *file = fopen(fileName, "r");
    if (file == NULL)
    {
        perror("Errore nell'apertura del file CSV");
        return EXIT_FAILURE;
    }

    int N = 0;
    char line[1024];
    while (fgets(line, sizeof(line), file))
    {
        N++;
    }

    double *h_data = (double *)malloc(N * d * sizeof(double));
    if (h_data == NULL)
    {
        perror("Errore nell'allocazione della memoria per i dati");
        fclose(file);
        return EXIT_FAILURE;
    }

    rewind(file);
    int i = 0;
    while (fgets(line, sizeof(line), file))
    {
        char *token = strtok(line, ",");
        for (int j = 0; j < d; ++j)
        {
            if (token != NULL)
            {
                h_data[i * d + j] = atof(token);
                token = strtok(NULL, ",");
            }
            else
            {
                fprintf(stderr, "Errore nella lettura del file CSV alla riga %d\n", i + 1);
                free(h_data);
                fclose(file);
                return EXIT_FAILURE;
            }
        }
        i++;
    }
    fclose(file);

    int totalThreads = (N + dataPerThread - 1) / dataPerThread;
    int numBlocks = (N + threadsPerBlock * dataPerThread - 1) / (threadsPerBlock * dataPerThread);

    double *h_means = (double *)malloc(k * d * sizeof(double));
    // double* h_local_means = (double*)malloc(numBlocks* threadsPerBlock * k * d * sizeof(double));
    double *h_covMatrices = (double *)malloc(k * d * d * sizeof(double));
    double *h_weights = (double *)malloc(k * sizeof(double));
    double *prev_h_means = (double *)malloc(k * d * sizeof(double));
    double *prev_h_covMatrices = (double *)malloc(k * d * d * sizeof(double));
    double *prev_h_weights = (double *)malloc(k * sizeof(double));
    // double* h_local_weights = (double*)malloc(threadsPerBlock * numBlocks * k * sizeof(double));
    // double* h_local_cov_matrixes = (double*)malloc(threadsPrBlock * numBlocks * k * d * d * sizeof(double));

    initializeMeans(h_data, h_means, N, d, k);

    for (int i = 0; i < k; ++i)
    {
        h_weights[i] = 1.0 / k;
        for (int j = 0; j < d; ++j)
        {
            for (int l = 0; l < d; ++l)
            {
                h_covMatrices[i * d * d + j * d + l] = (j == l) ? 1.0 : 0.0;
            }
        }
    }

    // print intial means
    printf("Means:\n");
    for (int i = 0; i < k; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            printf("%.9f ", h_means[i * d + j]);
        }
        printf("\n");
    }

    double *d_data, *d_means, *d_covMatrices, *d_weights, *d_responsibilities, *d_invCovMatrices, *d_determinants, *d_local_means, *d_local_weights, *d_local_cov_matrixes;
    CUDA_CHECK(cudaMalloc(&d_data, N * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_means, k * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_covMatrices, k * d * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_weights, k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_responsibilities, N * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_invCovMatrices, k * d * d * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_determinants, k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_means, k * d * threadsPerBlock * numBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_weights, k * threadsPerBlock * numBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_local_cov_matrixes, k * d * d * threadsPerBlock * numBlocks * sizeof(double)));

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

    printf("Numero di blocchi: %d\n", numBlocks);
    printf("Numero di thread per blocco: %d\n", threadsPerBlock);
    printf("Numero totale di thread: %d\n", totalThreads);
    printf("Numero di dati per thread: %d\n", dataPerThread);

    for (int iter = 0; iter < maxIter; ++iter)
    {
        // printf("Iterazione %d\n", iter + 1);

        computeInverseMatrices(handle, d_covMatrices, d, k, d_invCovMatrices, d_determinants);
        cudaDeviceSynchronize();

        computeResponsibilities<<<numBlocks, threadsPerBlock /* (N + 255) / 256, 256 */>>>(
            d_data, d_means, d_invCovMatrices, d_determinants, d_weights,
            d_responsibilities, d_local_means, d_local_weights, d, k, N);
        cudaDeviceSynchronize();

        // copia means e weights e covmatrix h nei valori precedenti
        memcpy(prev_h_means, h_means, k * d * sizeof(double));
        memcpy(prev_h_covMatrices, h_covMatrices, k * d * d * sizeof(double));
        memcpy(prev_h_weights, h_weights, k * sizeof(double));

        // copy back local means and weights
        //  double* h_local_means = (double*)malloc(numBlocks * threadsPerBlock * k * d * sizeof(double));
        //  double* h_local_weights = (double*)malloc(numBlocks * threadsPerBlock * k * sizeof(double));
        //  CUDA_CHECK(cudaMemcpy(h_local_means, d_local_means, numBlocks * threadsPerBlock * k * d * sizeof(double), cudaMemcpyDeviceToHost));
        //  CUDA_CHECK(cudaMemcpy(h_local_weights, d_local_weights, numBlocks * threadsPerBlock * k * sizeof(double), cudaMemcpyDeviceToHost));
        // Massimo numero di thread supportato per blocco
        int maxSharedMemory = 49152; // 48 KB

        int sharedMemPerThread = d * sizeof(double);
        int reduNumThreads = threadsPerBlock;

        // Calcolo del numero massimo di thread che non eccede la memoria condivisa
        if (sharedMemPerThread > 0)
        {
            reduNumThreads = threadsPerBlock < (maxSharedMemory / sharedMemPerThread)
                                 ? threadsPerBlock
                                 : (maxSharedMemory / sharedMemPerThread);
        }
        printf("Numero di thread per blocco reduce 1: %d\n", reduNumThreads);

        reduceWeightMean<<<k, threadsPerBlock>>>(d_local_means, d_local_weights, d_means, d_weights, d, k, k * threadsPerBlock * numBlocks, (totalThreads + reduNumThreads - 1) / reduNumThreads);
        cudaDeviceSynchronize();

        // double* h_means2 = (double*)malloc(k * d * sizeof(double));
        // double* h_weights2 = (double*)malloc(k * sizeof(double));

        // //print means2 and weights2
        // printf("Means2 GPU:\n");
        // for (int i = 0; i < k; ++i) {
        //     for (int j = 0; j < d; ++j) {
        //         printf("%f ", h_means2[i * d + j]);
        //     }
        //     printf("\n");
        // }

        // printf("Weights2 GPU:\n");
        // for (int i = 0; i < k; ++i) {
        //     printf("%f ", h_weights2[i]);
        // }

        // free(h_means2);
        // free(h_weights2);

        // cudaError_t err = cudaGetLastError();
        //     if (err != cudaSuccess) {
        //         printf("CUDA error: %s\n", cudaGetErrorString(err));
        //     }

        // sum local means and weights
        //  for (int i = 0; i < k; ++i) {
        //      for (int j = 0; j < d; ++j) {
        //          h_means[i * d + j] = 0.0;
        //      }
        //      h_weights[i] = 0.0;
        //  }

        //  for (int i = 0; i < numBlocks * threadsPerBlock; ++i) {
        //      for (int j = 0; j < k; ++j) {
        //          for (int l = 0; l < d; ++l) {
        //              h_means[j * d + l] += h_local_means[i * k * d + j * d + l];
        //          }
        //          h_weights[j] += h_local_weights[i * k + j];
        //      }
        //  }

        //   for (int i = 0; i < k; ++i) {
        //       for (int j = 0; j < d; ++j) {
        //          h_means[i * d + j] /= h_weights[i];
        //       }
        //       // h_weights[i] /= N;
        //  }

        //  // print means
        //     printf("Means CPU:\n");
        //     for (int i = 0; i < k; ++i) {
        //         for (int j = 0; j < d; ++j) {
        //             printf("%f ", h_means[i * d + j]);
        //         }
        //         printf("\n");
        //     }

        //     printf("Weights CPU:\n");
        //     for (int i = 0; i < k; ++i) {
        //         printf("%f ", h_weights[i]);
        //     }

        // CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(double), cudaMemcpyDeviceToHost));
        // free(h_local_means);

        // // print responsabilities
        // double* h_responsibilities = (double*)malloc(N * k * sizeof(double));
        // CUDA_CHECK(cudaMemcpy(h_responsibilities, d_responsibilities, N * k * sizeof(double), cudaMemcpyDeviceToHost));

        // printf("Responsabilities:\n");
        // for (int i = 0; i < N; ++i) {
        //     for (int j = 0; j < k; ++j) {
        //         printf("%f ", h_responsibilities[i * k + j]);
        //     }
        //     printf("\n");
        // }
        // printf("\n\n\n");

        // free(h_responsibilities);

        mStep<<<numBlocks, threadsPerBlock>>>(
            d_data, d_responsibilities, d_means, d_local_cov_matrixes, d, k, N);
        cudaDeviceSynchronize();
        checkCudaError("mStep");

        // copy back local cov matrixes
        // double* h_local_cov_matrixes = (double*)malloc(numBlocks * threadsPerBlock * k * d * d * sizeof(double));
        // CUDA_CHECK(cudaMemcpy(h_local_cov_matrixes, d_local_cov_matrixes, numBlocks * threadsPerBlock * k * d * d * sizeof(double), cudaMemcpyDeviceToHost));

        // printf("Local cov matrixes:\n");
        /*  for (int i = 0; i < numBlocks * threadsPerBlock; ++i) {
            for (int cluster = 0; cluster < k; ++cluster) {
                for (int j = 0; j < d; ++j) {
                    for (int l = 0; l < d; ++l) {
                        printf("%f ", h_local_cov_matrixes[i * k * d * d + cluster * d * d + j * d + l]);
                    }
                    printf("\n");
                }
                printf("\n\n");
            }
        }
        printf("\n\n\n"); */
        // Define the number of threads per block and the number of blocks
        // Memoria condivisa massima (48 KB per molte GPU moderne)

        sharedMemPerThread = d * d * sizeof(double);
        reduNumThreads = threadsPerBlock;

        // Calcolo del numero massimo di thread che non eccede la memoria condivisa
        if (sharedMemPerThread > 0)
        {
            reduNumThreads = threadsPerBlock < (maxSharedMemory / sharedMemPerThread)
                                 ? threadsPerBlock
                                 : (maxSharedMemory / sharedMemPerThread);
        }

        printf("Numero di thread per blocco reduce 2: %d\n", reduNumThreads);

        // Calculate the size of the shared memory needed
        unsigned int sharedMemSize = reduNumThreads * d * d * sizeof(double);

        // Kernel call
        reduceCovMatrices<<<k, reduNumThreads, sharedMemSize>>>(
            d_local_cov_matrixes, d_covMatrices, d_weights, d, k, k * threadsPerBlock * numBlocks, N, ((totalThreads + reduNumThreads - 1) / reduNumThreads));

        // Synchronize the device
        cudaDeviceSynchronize();
        checkCudaError("reduceCovMatrices");
        // checkCudaError("reduceCovMatrices");
        //         //int sharedMemSize = threadsPerBlock * d * d * sizeof(double);
        //         reduceCoMatrix<<<k, 32>>>(
        //                 d_local_cov_matrixes, d_means, d_weights, d_covMatrices, d, k, totalThreads * k, ((totalThreads + 32 - 1) / 32));
        //         cudaDeviceSynchronize();
        //         checkCudaError("reduceCoMatrix");

        // // copy back global cov matrixes
        CUDA_CHECK(cudaMemcpy(h_means, d_means, k * d * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_weights, d_weights, k * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_covMatrices, d_covMatrices, k * d * d * sizeof(double), cudaMemcpyDeviceToHost));

        // // reset global cov matrixes
        // for (int i = 0; i < k; ++i) {
        //     for (int j = 0; j < d; ++j) {
        //         for (int l = 0; l < d; ++l) {
        //             h_covMatrices[i * d * d + j * d + l] = 0.0;
        //         }
        //     }
        // }

        // // sum local cov matrixes
        // for (int i = 0; i < numBlocks * threadsPerBlock; ++i) {
        //     for (int cluster = 0; cluster < k; ++cluster) {
        //         for (int j = 0; j < d; ++j) {
        //             for (int l = 0; l < d; ++l) {
        //                 h_covMatrices[cluster * d * d + j * d + l] += h_local_cov_matrixes[i * k * d * d + cluster * d * d + j * d + l]; //[cluster][j][l] += [i][cluster][j][l]
        //             }
        //         }
        //     }
        // }

        // // normalize global cov matrixes
        // for (int i = 0; i < k; ++i) {
        //     for (int j = 0; j < d; ++j) {
        //         for (int l = 0; l < d; ++l) {
        //             h_covMatrices[i * d * d + j * d + l] /= h_weights[i];
        //             if(j == l){
        //                 h_covMatrices[i * d * d + j * d + l] += 0.0001;
        //             }
        //         }
        //     }
        //     h_weights[i] /= N;
        // }

        // check if prev local means and weights and covMatrix the diff is less than epsilon
        maxChange = 0.0;
        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < d; ++j)
            {
                double diff = fabs(h_means[i * d + j] - prev_h_means[i * d + j]);
                if (diff > maxChange)
                {
                    maxChange = diff;
                }
            }
        }

        for (int i = 0; i < k; ++i)
        {
            for (int j = 0; j < d; ++j)
            {
                for (int l = 0; l < d; ++l)
                {
                    double diff = fabs(h_covMatrices[i * d * d + j * d + l] - prev_h_covMatrices[i * d * d + j * d + l]);
                    if (diff > maxChange)
                    {
                        maxChange = diff;
                    }
                }
            }
        }

        for (int i = 0; i < k; ++i)
        {
            double diff = fabs(h_weights[i] - prev_h_weights[i]);
            if (diff > maxChange)
            {
                maxChange = diff;
            }
        }

        // CUDA_CHECK(cudaMemcpy(d_covMatrices, h_covMatrices, k * d * d * sizeof(double), cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_weights, h_weights, k * sizeof(double), cudaMemcpyHostToDevice));

        // free(h_local_cov_matrixes);
        // free(h_local_weights);

        // if (maxChange < epsilon) {
        //     printf("Convergenza raggiunta dopo %d iterazioni\n", iter + 1);
        //     break;
        // }
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
    for (int i = 0; i < k; ++i)
    {
        printf("Cluster %d:\n", i + 1);
        printf("Mean: ");
        for (int j = 0; j < d; ++j)
        {
            printf("%f ", h_means[i * d + j]);
        }
        printf("\nCovariance Matrix:\n");
        for (int j = 0; j < d; ++j)
        {
            for (int l = 0; l < d; ++l)
            {
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
    cudaFree(d_invCovMatrices);
    cudaFree(d_determinants);
    cudaFree(d_local_means);
    cudaFree(d_local_weights);
    cudaFree(d_local_cov_matrixes);
    free(prev_h_means);
    free(prev_h_covMatrices);
    free(prev_h_weights);
    free(h_data);
    free(h_means);
    free(h_covMatrices);
    free(h_weights);

    cublasDestroy(handle);

    return 0;
}