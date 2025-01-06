#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <chrono> // Per i timer

using namespace Eigen;
using namespace std;

// Funzione per leggere i dati da un file CSV
std::vector<VectorXd> loadData(const std::string& filename, int& numFeatures) {
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
        } else if (values.size() != numFeatures) {
            throw std::runtime_error("Numero di feature incoerente nei dati");
        }

        VectorXd point = VectorXd::Map(values.data(), values.size());
        data.push_back(point);
    }
    file.close();
    return data;
}

// Funzione di inizializzazione casuale dei parametri del GMM
void initializeParameters(std::vector<VectorXd>& means, std::vector<MatrixXd>& covariances,
                          std::vector<double>& weights, int k, const std::vector<VectorXd>& data, int numFeatures) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    for (int i = 0; i < k; ++i) {
        means[i] = data[dis(gen)];
        covariances[i] = MatrixXd::Identity(numFeatures, numFeatures);
        weights[i] = 1.0f / k;
    }
}

// Calcola la densità di probabilità gaussiana
double gaussian(const VectorXd& x, const VectorXd& mean, const MatrixXd& covariance, MatrixXd covInv, double det) {
    VectorXd diff = x - mean;
    double exponent = -0.5 * diff.transpose() * covInv * diff;
    double norm = 1.0 / (std::pow(2 * M_PI, x.size() / 2.0) * std::sqrt(det));
    return norm * std::exp(exponent);
}

// E-step
MatrixXd computeResponsibilities(const std::vector<VectorXd>& data, const std::vector<VectorXd>& means,
                                  const std::vector<MatrixXd>& covariances, const std::vector<double>& weights, int k) {
    MatrixXd responsibilities(data.size(), k);
    // calculate covInv
    // calculate det
    MatrixXd covInv[k];
    double det[k];
    for (int j = 0; j < k; ++j) {
        covInv[j] = covariances[j].inverse();
        det[j] = covariances[j].determinant();
    }
    for (size_t i = 0; i < data.size(); ++i) {
        double sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            responsibilities(i, j) = weights[j] * gaussian(data[i], means[j], covariances[j], covInv[j], det[j]);
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
     // std::cout << "Responsabilità: \n" << responsibilities << std::endl;
    return responsibilities;
}

// M-step
void updateParameters(const std::vector<VectorXd>& data, const MatrixXd& responsibilities, std::vector<VectorXd>& means,
                      std::vector<MatrixXd>& covariances, std::vector<double>& weights, int k) {
    size_t n = data.size();
    int numFeatures = data[0].size();

    for (int j = 0; j < k; ++j) {
        double responsibilitySum = responsibilities.col(j).sum();

        // Aggiorna il peso
        weights[j] = responsibilitySum / n;

        // Aggiorna la media
        VectorXd newMean = VectorXd::Zero(numFeatures);
        for (size_t i = 0; i < n; ++i) {
            newMean += responsibilities(i, j) * data[i];
        }
        newMean /= responsibilitySum;
        means[j] = newMean;

        // Aggiorna la matrice di covarianza
        MatrixXd newCovariance = MatrixXd::Zero(numFeatures, numFeatures);
        for (size_t i = 0; i < n; ++i) {
            VectorXd diff = data[i] - means[j];
            newCovariance += responsibilities(i, j) * (diff * diff.transpose());
        }
        newCovariance /= responsibilitySum;
        covariances[j] = newCovariance;
    }
}

/* void updateParameters(const std::vector<VectorXd>& data, const MatrixXd& responsibilities, 
                      std::vector<VectorXd>& means, std::vector<MatrixXd>& covariances, 
                      std::vector<double>& weights, int k) {
    size_t n = data.size();
    int numFeatures = data[0].size();

    // Pre-allocazione di variabili per ridurre le allocazioni dinamiche
    MatrixXd dataMatrix(n, numFeatures);
    for (size_t i = 0; i < n; ++i) {
        dataMatrix.row(i) = data[i];
    }

    for (int j = 0; j < k; ++j) {
        // Somma delle responsabilità per il cluster j
        double responsibilitySum = responsibilities.col(j).sum();

        // Aggiorna il peso
        weights[j] = responsibilitySum / n;

        // Calcolo vettorializzato della media
        VectorXd newMean = (responsibilities.col(j).transpose() * dataMatrix).transpose() / responsibilitySum;
        means[j] = newMean;

        // Calcolo vettorializzato della covarianza
        MatrixXd centered = dataMatrix.rowwise() - newMean.transpose();
        MatrixXd weightedCentered = (responsibilities.col(j).asDiagonal() * centered);
        MatrixXd newCovariance = (weightedCentered.transpose() * centered) / responsibilitySum;
        covariances[j] = newCovariance;
    }
} */

int main() {
    const std::string filename = "../data/1M.csv";
    const int k = 5; // Numero di cluster
    const int maxIter = 5; // Numero massimo di iterazioni
    int numFeatures = 0;

    try {
        auto startTotal = std::chrono::high_resolution_clock::now();
        // Caricamento dati
        std::vector<VectorXd> data = loadData(filename, numFeatures);

        // Inizializzazione dei parametri
        std::vector<VectorXd> means(k, VectorXd(numFeatures));
        std::vector<MatrixXd> covariances(k, MatrixXd(numFeatures, numFeatures));
        std::vector<double> weights(k);
        initializeParameters(means, covariances, weights, k, data, numFeatures);
        /* for (int j = 0; j < k; ++j) {
            std::cout << "Cluster " << j + 1 << ":\n";
            std::cout << "Mean:\n" << means[j] << "\n";
            std::cout << "Covariance:\n" << covariances[j] << "\n";
            std::cout << "Weight: " << weights[j] << "\n\n";
        } */
        // Timer per il calcolo del tempo totale

        // Iterazioni dell'algoritmo EM
        for (int iter = 0; iter < maxIter; ++iter) {
            auto startIter = std::chrono::high_resolution_clock::now();

            MatrixXd responsibilities = computeResponsibilities(data, means, covariances, weights, k);
            auto endEStep = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> eStepTime = endEStep - startIter;
            std::cout << "Tempo E-step: " << eStepTime.count() << " secondi\n";

            updateParameters(data, responsibilities, means, covariances, weights, k);
            auto endMStep = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> mStepTime = endMStep - endEStep;

            std::cout << "Tempo M-step: " << mStepTime.count() << " secondi\n";

            auto endIter = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> iterTime = endIter - startIter;

            std::cout << "Tempo iterazione " << iter + 1 << ": " << iterTime.count() << " secondi\n";
           /*  for (int j = 0; j < k; ++j) {
                std::cout << "Cluster " << j + 1 << ":\n";
                std::cout << "Mean:\n" << means[j] << "\n";
                std::cout << "Covariance:\n" << covariances[j] << "\n";
                std::cout << "Weight: " << weights[j] << "\n\n";
            } */
        }

        auto endTotal = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> totalTime = endTotal - startTotal;

        std::cout << "\nTempo totale: " << totalTime.count() << " secondi\n";

        // Stampa dei risultati
         for (int j = 0; j < k; ++j) {
            std::cout << "Cluster " << j + 1 << ":\n";
            std::cout << "Mean:\n" << means[j] << "\n";
            std::cout << "Covariance:\n" << covariances[j] << "\n";
            std::cout << "Weight: " << weights[j] << "\n\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Errore: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
