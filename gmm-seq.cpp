#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

class MultivariateGaussian {
private:
    VectorXd mean;
    MatrixXd covariance;
    double normalization_constant;
    
    void computeNormalizationConstant() {
        double det = covariance.determinant();
        int d = mean.size();
        normalization_constant = 1.0 / (std::pow(2 * M_PI, d/2.0) * std::sqrt(det));
    }

public:
    MultivariateGaussian(const VectorXd& mu, const MatrixXd& sigma) 
        : mean(mu), covariance(sigma) {
        computeNormalizationConstant();
    }
    
    double pdf(const VectorXd& x) const {
        VectorXd diff = x - mean;
        double exponent = -0.5 * diff.transpose() * covariance.inverse() * diff;
        return normalization_constant * std::exp(exponent);
    }
};

class GMM {
private:
    std::vector<MultivariateGaussian> components;
    std::vector<double> weights;
    int n_components;
    int n_dimensions;
    
    MatrixXd responsibilities;
    std::vector<VectorXd> means;
    std::vector<MatrixXd> covariances;

public:
    GMM(int k, int d) : n_components(k), n_dimensions(d) {
        weights.resize(k, 1.0/k);
        means.resize(k);
        covariances.resize(k);
        
        // Initialize with random means and identity covariances
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0, 1);
        
        for (int i = 0; i < k; i++) {
            VectorXd mean = VectorXd::Zero(d);
            for (int j = 0; j < d; j++) {
                mean(j) = dist(gen);
            }
            means[i] = mean;
            covariances[i] = MatrixXd::Identity(d, d);
            components.emplace_back(mean, MatrixXd::Identity(d, d));
        }
    }
    
    void fit(const std::vector<VectorXd>& data, int max_iter=100, double tol=1e-3) {
        int n_samples = data.size();
        responsibilities = MatrixXd::Zero(n_samples, n_components);
        
        double prev_log_likelihood = -INFINITY;
        
        for (int iter = 0; iter < max_iter; iter++) {
            // E-step
            for (int i = 0; i < n_samples; i++) {
                double total = 0;
                for (int j = 0; j < n_components; j++) {
                    responsibilities(i, j) = weights[j] * components[j].pdf(data[i]);
                    total += responsibilities(i, j);
                }
                responsibilities.row(i) /= total;
            }
            
            // M-step
            for (int j = 0; j < n_components; j++) {
                VectorXd new_mean = VectorXd::Zero(n_dimensions);
                MatrixXd new_cov = MatrixXd::Zero(n_dimensions, n_dimensions);
                double resp_sum = 0;
                
                for (int i = 0; i < n_samples; i++) {
                    double resp = responsibilities(i, j);
                    new_mean += resp * data[i];
                    resp_sum += resp;
                }
                new_mean /= resp_sum;
                
                for (int i = 0; i < n_samples; i++) {
                    VectorXd diff = data[i] - new_mean;
                    new_cov += responsibilities(i, j) * diff * diff.transpose();
                }
                new_cov /= resp_sum;
                
                // Update parameters
                weights[j] = resp_sum / n_samples;
                means[j] = new_mean;
                covariances[j] = new_cov;
                components[j] = MultivariateGaussian(new_mean, new_cov);
            }
            
            // Check convergence
            double log_likelihood = 0;
            for (int i = 0; i < n_samples; i++) {
                double sample_likelihood = 0;
                for (int j = 0; j < n_components; j++) {
                    sample_likelihood += weights[j] * components[j].pdf(data[i]);
                }
                log_likelihood += std::log(sample_likelihood);
            }
            
            if (std::abs(log_likelihood - prev_log_likelihood) < tol) {
                break;
            }
            prev_log_likelihood = log_likelihood;
        }
    }
    
    VectorXd predict_proba(const VectorXd& x) const {
        VectorXd probs(n_components);
        for (int j = 0; j < n_components; j++) {
            probs(j) = weights[j] * components[j].pdf(x);
        }
        probs /= probs.sum();
        return probs;
    }
};