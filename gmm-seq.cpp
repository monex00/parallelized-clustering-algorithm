#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <random>
#include <chrono> // Per i timer

using namespace Eigen;

// Funzione per leggere i dati da un file CSV
std::vector<Vector2f> loadData(const std::string& filename) {
    std::vector<Vector2f> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Impossibile aprire il file CSV");
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        float x, y;
        if (std::getline(ss, value, ',')) {
            x = std::stof(value);
        }
        if (std::getline(ss, value, ',')) {
            y = std::stof(value);
        }
        data.emplace_back(Vector2f(x, y));
    }
    file.close();
    return data;
}

// Funzione di inizializzazione casuale dei parametri del GMM
void initializeParameters(std::vector<Vector2f>& means, std::vector<Matrix2f>& covariances,
                          std::vector<float>& weights, int k, const std::vector<Vector2f>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, data.size() - 1);

    for (int i = 0; i < k; ++i) {
        means[i] = data[dis(gen)];
        covariances[i] = Matrix2f::Identity();
        weights[i] = 1.0f / k;
    }
}

// Calcola la densità di probabilità gaussiana
float gaussian(const Vector2f& x, const Vector2f& mean, const Matrix2f& covariance) {
    Matrix2f covInv = covariance.inverse();
    float det = covariance.determinant();
    Vector2f diff = x - mean;

    float exponent = -0.5f * diff.transpose() * covInv * diff;
    float norm = 1.0f / (std::pow(2 * M_PI, 2 / 2.0f) * std::sqrt(det));
    return norm * std::exp(exponent);
}

// E-step
MatrixXf computeResponsibilities(const std::vector<Vector2f>& data, const std::vector<Vector2f>& means,
                                  const std::vector<Matrix2f>& covariances, const std::vector<float>& weights, int k) {
    MatrixXf responsibilities(data.size(), k);
    for (size_t i = 0; i < data.size(); ++i) {
        float sum = 0.0f;
        for (int j = 0; j < k; ++j) {
            responsibilities(i, j) = weights[j] * gaussian(data[i], means[j], covariances[j]);
            sum += responsibilities(i, j);
        }
        responsibilities.row(i) /= sum; // Normalizza
    }
    return responsibilities;
}

// M-step
void updateParameters(const std::vector<Vector2f>& data, const MatrixXf& responsibilities, std::vector<Vector2f>& means,
                      std::vector<Matrix2f>& covariances, std::vector<float>& weights, int k) {
    size_t n = data.size();

    for (int j = 0; j < k; ++j) {
        float responsibilitySum = responsibilities.col(j).sum();

        // Aggiorna il peso
        weights[j] = responsibilitySum / n;

        // Aggiorna la media
        Vector2f newMean = Vector2f::Zero();
        for (size_t i = 0; i < n; ++i) {
            newMean += responsibilities(i, j) * data[i];
        }
        newMean /= responsibilitySum;
        means[j] = newMean;

        // Aggiorna la matrice di covarianza
        Matrix2f newCovariance = Matrix2f::Zero();
        for (size_t i = 0; i < n; ++i) {
            Vector2f diff = data[i] - means[j];
            newCovariance += responsibilities(i, j) * (diff * diff.transpose());
        }
        newCovariance /= responsibilitySum;
        covariances[j] = newCovariance;
    }
}

int main() {
    const std::string filename = "data.csv";
    const int k = 2; // Numero di cluster
    const int maxIter = 1000; // Numero massimo di iterazioni

    try {
        // Caricamento dati
        std::vector<Vector2f> data = loadData(filename);

        // Inizializzazione dei parametri
        std::vector<Vector2f> means(k);
        std::vector<Matrix2f> covariances(k);
        std::vector<float> weights(k);
        initializeParameters(means, covariances, weights, k, data);

        // Timer per il calcolo del tempo totale
        auto startTotal = std::chrono::high_resolution_clock::now();

        // Iterazioni dell'algoritmo EM
        for (int iter = 0; iter < maxIter; ++iter) {
            // Timer per il calcolo del tempo di iterazione
            auto startIter = std::chrono::high_resolution_clock::now();

            MatrixXf responsibilities = computeResponsibilities(data, means, covariances, weights, k);
            updateParameters(data, responsibilities, means, covariances, weights, k);

            // Fine timer per l'iterazione
            auto endIter = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> iterTime = endIter - startIter;

            std::cout << "Tempo iterazione " << iter + 1 << ": " << iterTime.count() << " secondi\n";
        }

        // Fine timer per il tempo totale
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
