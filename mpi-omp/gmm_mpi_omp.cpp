#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <chrono>
#include <omp.h>
#include <mpi.h>
#include <iomanip> 


using namespace Eigen;
using namespace std;

std::vector<VectorXd> loadData(const std::string &filename, int &numFeatures) {
    std::vector<VectorXd> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Impossibile aprire il file CSV");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> values;

        while (std::getline(ss, value, ',')) {
            values.push_back(std::stof(value));
        }

        if (data.empty()) {
            numFeatures = values.size();
        }

        VectorXd point = VectorXd::Map(values.data(), values.size());
        data.push_back(point);
    }
    file.close();
    return data;
}

void initializeParameters(std::vector<VectorXd> &means, std::vector<MatrixXd> &covariances,
                          std::vector<double> &weights, int k, const std::vector<VectorXd> &data, int numFeatures) {
  

    //somma le feature di tutti i dati
    /* VectorXd sum = VectorXd::Zero(numFeatures);
    for (int i = 0; i < data.size(); ++i) {
        sum += data[i];
    }

    //fai la media e mettila come media del cluster
    for (int i = 0; i < k; ++i) {
        std::cout << "Sum: \n" << sum << std::endl;
        std::cout << "Data Size: " << data.size() << std::endl;


        means[i] = (sum / data.size()) +  (VectorXd::Random(numFeatures) + VectorXd::Ones(numFeatures)) * 0.5;
        std::cout << "Means1[" << i << "]: \n" << (sum / data.size()) << std::endl;
        std::cout << "Means2[" << i << "]: \n" << means[i] << std::endl;
        covariances[i] = MatrixXd::Identity(numFeatures, numFeatures);
        weights[i] = 1.0 / k;
    } */

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    for (int i = 0; i < k; ++i) {
        means[i] = data[dis(gen)];
        covariances[i] = MatrixXd::Identity(numFeatures, numFeatures);
        weights[i] = 1.0f / k;
    }
    
}

double gaussian(const VectorXd &x, const VectorXd &mean, const MatrixXd &covariance, MatrixXd covInv, double det) {
   /*  MatrixXd covInv = covariance.inverse();
    double det = covariance.determinant(); */
    VectorXd diff = x - mean;


    double exponent = -0.5 * diff.transpose() * covInv * diff;
    double norm = 1.0 / (std::pow(2 * M_PI, x.size() / 2.0) * std::sqrt(det));
   
    return norm * std::exp(exponent);
}

MatrixXd computeResponsibilities(const std::vector<VectorXd> &data, const std::vector<VectorXd> &means,
                                  const std::vector<MatrixXd> &covariances, const std::vector<double> &weights, int k,int rank, int size) {
    MatrixXd responsibilities(data.size(), k);
    std::vector<MatrixXd> covInv(k, MatrixXd(data[0].size(), data[0].size()));
    std::vector<double> det(k);

    for (int j = 0; j < k; ++j) {
        if(rank == 0) {
            covInv[j] = covariances[j].inverse();
            det[j] = covariances[j].determinant();
        }
        MPI_Bcast(covInv[j].data(), data[0].size() * data[0].size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&det[j], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); ++i) {
        double sum = 0.0;
        //std::cout << "Rank: " << rank << " Sum Prima: " << sum << std::endl;
        //#pragma omp parallel for reduction(+:sum) //a quanto pare per quanto consiglino di non usare due for paralleli interni qua, sembra andare meglio
        for (int j = 0; j < k; ++j) {
            double gaussianValue = gaussian(data[i], means[j], covariances[j], covInv[j], det[j]);
            double weightedGaussian = weights[j] * gaussianValue;

            responsibilities(i, j) = weightedGaussian;

            sum += responsibilities(i, j);
        }
        

        if(sum < 1e-10){
            for (int j = 0; j < k; ++j) {
                responsibilities(i, j) = 1.0 / k;
            }
        }else{
            responsibilities.row(i) /= sum;
        }
    }
    return responsibilities;
}

/* void updateParameters(const std::vector<VectorXd> &localData,int globalDataSize, const MatrixXd &responsibilities,
                      std::vector<VectorXd> &means, std::vector<MatrixXd> &covariances,
                      std::vector<double> &weights, int k, int rank, int size, int numFeatures) {
    std::vector<double> localSums(k, 0.0);
    std::vector<VectorXd> localMeans(k, VectorXd::Zero(numFeatures));
    std::vector<MatrixXd> localCovariances(k, MatrixXd::Zero(numFeatures, numFeatures));
    
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < localData.size(); ++i) {
        for (int j = 0; j < k; ++j) {
            localSums[j] += responsibilities(i, j);
            localMeans[j] += responsibilities(i, j) * localData[i];
        }
    }

    std::vector<double> globalSums(k, 0.0);
    std::vector<VectorXd> globalMeans(k, VectorXd::Zero(numFeatures));
    std::vector<MatrixXd> globalCovariances(k, MatrixXd::Zero(numFeatures, numFeatures));
    MPI_Reduce(localSums.data(), globalSums.data(), k, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    for (int j = 0; j < k; ++j) {
        MPI_Reduce(localMeans[j].data(), globalMeans[j].data(), numFeatures, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        for (int j = 0; j < k; ++j) {
            // stampa globalMeans[j]
            // std::cout << "GlobalMeans[" << j << "]: " << globalMeans[j] << std::endl;
            weights[j] = globalSums[j] / globalDataSize;
            means[j] = globalMeans[j] / globalSums[j];
        }
    }

    // ivia a tutti la nuova media
    for (int j = 0; j < k; ++j) {
        MPI_Bcast(means[j].data(), numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // invia i pesi
    MPI_Bcast(weights.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ognuno calcola la nuova covarianza
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < localData.size(); ++i) {
        for (int j = 0; j < k; ++j) {
            VectorXd diff = localData[i] - means[j];
            localCovariances[j] += responsibilities(i, j) * (diff * diff.transpose());
        }
    }

    // riduce le covarianze
    for (int j = 0; j < k; ++j) {
        MPI_Reduce(localCovariances[j].data(), globalCovariances[j].data(), numFeatures * numFeatures, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            globalCovariances[j] /= globalSums[j];
            // aggiungi 0.0001 alla diagonale
            covariances[j] = globalCovariances[j] + 0.0001 * MatrixXd::Identity(numFeatures, numFeatures);

            
        }
        MPI_Bcast(covariances[j].data(), numFeatures * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

} */

void updateParameters(const std::vector<VectorXd> &localData, int globalDataSize, 
                      const MatrixXd &responsibilities, std::vector<VectorXd> &means, 
                      std::vector<MatrixXd> &covariances, std::vector<double> &weights, 
                      int k, int rank, int size, int numFeatures) {
    // Preallocazione per somme locali e medie
    std::vector<double> localSums(k, 0.0);
    std::vector<VectorXd> localMeans(k, VectorXd::Zero(numFeatures));
    std::vector<MatrixXd> localCovariances(k, MatrixXd::Zero(numFeatures, numFeatures));

    // Converti localData in una matrice Eigen::MatrixXd
    MatrixXd dataMatrix(localData.size(), numFeatures);
    for (size_t i = 0; i < localData.size(); ++i) {
        dataMatrix.row(i) = localData[i];
    }

    // Calcolo parallelo delle somme e delle medie
    #pragma omp parallel for
    for (int j = 0; j < k; ++j) {
        for (size_t i = 0; i < localData.size(); ++i) {
            localSums[j] += responsibilities(i, j);
            localMeans[j] += responsibilities(i, j) * localData[i];
        }
    }

    // Riduzione MPI per calcolare le somme globali
    std::vector<double> globalSums(k, 0.0);
    std::vector<VectorXd> globalMeans(k, VectorXd::Zero(numFeatures));
    MPI_Allreduce(localSums.data(), globalSums.data(), k, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for (int j = 0; j < k; ++j) {
        MPI_Allreduce(localMeans[j].data(), globalMeans[j].data(), numFeatures, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // Aggiorna i parametri globali
    if (rank == 0) {
        for (int j = 0; j < k; ++j) {
            weights[j] = globalSums[j] / globalDataSize;
            means[j] = globalMeans[j] / globalSums[j];
        }
    }

    // Broadcast delle nuove medie
    for (int j = 0; j < k; ++j) {
        MPI_Bcast(means[j].data(), numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(weights.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calcolo parallelo delle covarianze
    #pragma omp parallel for
    for (int j = 0; j < k; ++j) {
        MatrixXd centered = dataMatrix.rowwise() - means[j].transpose();
        localCovariances[j] = (responsibilities.col(j).asDiagonal() * centered).transpose() * centered;
    }

    // Riduzione MPI per calcolare le covarianze globali
    std::vector<MatrixXd> globalCovariances(k, MatrixXd::Zero(numFeatures, numFeatures));
    for (int j = 0; j < k; ++j) {
        MPI_Allreduce(localCovariances[j].data(), globalCovariances[j].data(), 
                      numFeatures * numFeatures, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }

    // Aggiorna le covarianze
    if (rank == 0) {
        for (int j = 0; j < k; ++j) {
            covariances[j] = globalCovariances[j] / globalSums[j] + 1e-6 * MatrixXd::Identity(numFeatures, numFeatures);
        }
    }

    // Broadcast delle nuove covarianze
    for (int j = 0; j < k; ++j) {
        MPI_Bcast(covariances[j].data(), numFeatures * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    double start_time, end_time, elapsed_time;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    start_time = MPI_Wtime();

    const std::string filename = "../data/s3.csv";
    const int k = 15;        // Numero di cluster
    const int maxIter = 20; // Numero massimo di iterazioni
    int numFeatures = 0;    // Numero di caratteristiche dei dati
    std::vector<VectorXd> fullData;
    std::vector<VectorXd> localData;



    int totalDataSize = 0;
    double loadDataStart = MPI_Wtime();  
    if (rank == 0) {
        fullData = loadData(filename, numFeatures);
        totalDataSize = fullData.size();
    }
    double loadDataEnd = MPI_Wtime();
    /* if(rank == 0){
        
        std::cout << "Tempo di caricamento dati: " << loadDataEnd - loadDataStart << " secondi." << std::endl;
    } */
    double splitDataStart = MPI_Wtime();  
    // Broadcast delle informazioni base
    MPI_Bcast(&numFeatures, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&totalDataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcolo dimensioni locali
    int localSize = totalDataSize / size + (rank < totalDataSize % size ? 1 : 0);
    std::vector<int> sendCounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        sendCounts[i] = (totalDataSize / size + (i < totalDataSize % size ? 1 : 0)) * numFeatures;
        displs[i] = (i > 0 ? displs[i - 1] + sendCounts[i - 1] : 0);
    }

    // Buffer contiguo per il rank 0
    std::vector<double> flatData;
    if (rank == 0) {
        flatData.resize(totalDataSize * numFeatures);
        for (int i = 0; i < fullData.size(); ++i) {
            for (int j = 0; j < numFeatures; ++j) {
                flatData[i * numFeatures + j] = fullData[i](j);
            }
        }
    }

    // Buffer contiguo per i dati locali
    std::vector<double> localFlatData(localSize * numFeatures);

    // Scatterv dei dati
    MPI_Scatterv(rank == 0 ? flatData.data() : nullptr, sendCounts.data(), displs.data(),
                 MPI_DOUBLE, localFlatData.data(), localSize * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // Converti i dati ricevuti in Eigen::VectorXd
    localData.resize(localSize, VectorXd(numFeatures));
    for (int i = 0; i < localSize; ++i) {
        localData[i] = VectorXd(numFeatures);
        for (int j = 0; j < numFeatures; ++j) {
            localData[i](j) = localFlatData[i * numFeatures + j];
        }
    }


    double splitDataEnd = MPI_Wtime(); 
    /* if(rank == 0){
        std::cout << "Tempo di divisione dati: " << splitDataEnd - splitDataStart << " secondi." << std::endl;
    } */
    // Inizializzazione dei parametri
    double initParamsStart = MPI_Wtime();  
    std::vector<VectorXd> means(k, VectorXd(numFeatures));
    std::vector<MatrixXd> covariances(k, MatrixXd::Identity(numFeatures, numFeatures));
    std::vector<double> weights(k, 1.0 / k);

    if (rank == 0) {
        initializeParameters(means, covariances, weights, k, fullData, numFeatures);
        // Stampa dei parametri iniziali
       /*  std::cout << "Parametri iniziali:\n";
        for (int j = 0; j < k; ++j) {
            std::cout << "Cluster " << j + 1 << ":\n";
            std::cout << "Mean:\n" << means[j] << "\n";
            std::cout << "Covariance:\n" << covariances[j] << "\n";
            std::cout << "Weight: " << weights[j] << "\n\n";
        } */
    }

    double initParamsEnd = MPI_Wtime();
    /* if (rank == 0) {
        std::cout << "Tempo di inizializzazione parametri: " << initParamsEnd - initParamsStart << " secondi." << std::endl;
    } */
    // Broadcast iniziale dei parametri ai processi
    for (int j = 0; j < k; ++j) {
        MPI_Bcast(means[j].data(), numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(covariances[j].data(), numFeatures * numFeatures, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Bcast(weights.data(), k, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double totaleStep = 0; 
    double totalmstep = 0;
    MatrixXd responsibilities;
    // Iterazioni EM
    for (int iter = 0; iter < maxIter; ++iter) {
        double eStepStart = MPI_Wtime();  
        
        // Calcolo delle responsabilità
        responsibilities = computeResponsibilities(localData, means, covariances, weights, k, rank, size);

        //stampa delle responsabilità
         /* for (int i = 0; i < responsibilities.rows(); ++i) {
             std::cout << "Responsabilità " << i << ": ";
             for (int j = 0; j < responsibilities.cols(); ++j) {
                 std::cout << std::fixed << std::setprecision(3) << responsibilities(i, j) << " ";
             }   
             std::cout << std::endl;
         } */

        double eStepEnd = MPI_Wtime();

        totaleStep += eStepEnd - eStepStart;

        if (rank == 0) {
            std::cout << "Tempo E-step (Iterazione " << iter + 1 << "): " << eStepEnd - eStepStart << " secondi." << std::endl;
        }

        
        double mStepStart = MPI_Wtime();  
        
        // Aggiornamento dei parametri
        updateParameters(localData, totalDataSize, responsibilities, means, covariances, weights, k, rank, size, numFeatures);

        double mStepEnd = MPI_Wtime();

        totalmstep += mStepEnd - mStepStart;

        if (rank == 0) {
            std::cout << "Tempo M-step (Iterazione " << iter + 1 << "): " << mStepEnd- mStepStart << " secondi." << std::endl;
            /* for (int j = 0; j < k; ++j) {
                std::cout << "Cluster " << j + 1 << ":\n";
                std::cout << "Mean:\n" << means[j] << "\n";
                std::cout << "Covariance:\n" << covariances[j] << "\n";
                std::cout << "Weight: " << weights[j] << "\n\n";
            } */
            std::cout << "Tempo Iterazione " << iter + 1 << ": " << mStepEnd - eStepStart << " secondi." << std::endl;
        }
    }

    end_time = MPI_Wtime();

    // Calcola il tempo trascorso
    elapsed_time = end_time - start_time;

    // Stampa dei risultati finali
    if (rank == 0) {
        //std::cout << "Tempo totale di esecuzione: " << elapsed_time << " secondi." << std::endl;
        std::cout << elapsed_time << std::endl;

        std::cout << "Tempo totale E-step: " << totaleStep << " secondi." << std::endl;
        std::cout << "Tempo totale M-step: " << totalmstep << " secondi." << std::endl;
        std::cout << "Risultati finali:\n";
        for (int j = 0; j < k; ++j) {
            std::cout << "Cluster " << j + 1 << ":\n";
            std::cout << "Mean:\n" << means[j] << "\n";
            std::cout << "Covariance:\n" << covariances[j] << "\n";
            std::cout << "Weisght: " << weights[j] << "\n\n";
        }
    }
/*     if (rank == 0) {
        FILE* paramFile = fopen("../py/model_params.csv", "w");
        if (paramFile == NULL) {
            perror("Errore nella creazione del file dei parametri");
            return EXIT_FAILURE;
        }

        fprintf(paramFile, "Cluster,Feature,Mean,Covariance,Weight\n");
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < numFeatures; ++i) {
                fprintf(paramFile, "%d,%d,%f,", j, i, means[j](i));
                for (int k = 0; k < numFeatures; ++k) {
                    fprintf(paramFile, "%f", covariances[j](i, k));
                }
            }
        }

        fclose(paramFile);
        FILE* respFile = fopen("../py/responsibilities.csv", "w");
        if (respFile == NULL) {
            perror("Errore nella creazione del file delle responsabilità");
            return EXIT_FAILURE;
        }
        fprintf(respFile, "DataPoint,Cluster,Responsibility\n");

        for (size_t i = 0; i < localData.size(); ++i) {
            for (int j = 0; j < k; ++j) {
                fprintf(respFile, "%d,%d,%f\n", i, j, responsibilities(i, j));
            }
        }

        fclose(respFile);
    } */

    MPI_Finalize();





    return 0;
}


