import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd
import datetime

def create_sample_data(n_samples=1000000, n_features=10, n_clusters=5, random_state=42):
    """
    Crea dei dati di esempio per il clustering  
    """
    X, y = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=n_clusters,
                      cluster_std=0.5,
                      random_state=random_state)
    return X

def save_dataset(X, filename=None, include_timestamp=True):
    # Crea un DataFrame con nomi delle colonne automatici
    df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(X.shape[1])])
    
    # Genera il nome del file
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dataset_{timestamp}.csv"
    elif include_timestamp and not filename.endswith('.csv'):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{timestamp}.csv"
    elif not filename.endswith('.csv'):
        filename = f"{filename}.csv"
    
    # Salva il DataFrame
    df.to_csv(filename, index=False, header=False)
    print(f"Dataset salvato in: {filename}")
    print(f"Dimensioni: {X.shape}")
    return filename

def load_dataset(filename):
    df = pd.read_csv(filename)
    X = df.values
    print(f"Dataset caricato da: {filename}")
    print(f"Dimensioni: {X.shape}")
    return X

def fit_gmm(X, n_components=4, random_state=42):
    """
    Addestra un modello GMM sui dati
    """
    gmm = GaussianMixture(n_components=n_components,
                         covariance_type='full',
                         random_state=random_state)
    gmm.fit(X)
    return gmm

def print_gmm_parameters(gmm):
    """
    Stampa i parametri del modello GMM
    """
    print("\nParametri del modello GMM:")
    print("-" * 50)
    
    for i in range(gmm.n_components):
        print(f"\nCluster {i+1}:")
        print(f"Media:\n{gmm.means_[i]}")
        print("\nMatrice di Covarianza:")
        for row in gmm.covariances_[i]:
            print(' '.join(f'{x:.6f}' for x in row))
        print(f"Peso: {gmm.weights_[i]:.6f}")

def save_gmm_parameters(gmm, filename=None):
    """
    Salva i parametri del modello GMM su file CSV
    """
    if filename is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gmm_params_{timestamp}.csv"
    elif not filename.endswith('.csv'):
        filename = f"{filename}.csv"

    # Crea un dizionario per salvare tutti i parametri
    params_dict = {}
    
    for i in range(gmm.n_components):
        # Salva le medie
        for j, mean_val in enumerate(gmm.means_[i]):
            params_dict[f'cluster_{i+1}_mean_{j+1}'] = [mean_val]
        
        # Salva le covarianze
        for j, cov_row in enumerate(gmm.covariances_[i]):
            for k, cov_val in enumerate(cov_row):
                params_dict[f'cluster_{i+1}_cov_{j+1}_{k+1}'] = [cov_val]
        
        # Salva i pesi
        params_dict[f'cluster_{i+1}_weight'] = [gmm.weights_[i]]

    # Converti in DataFrame e salva
    params_df = pd.DataFrame(params_dict)
    params_df.to_csv(filename, index=False)
    print(f"Parametri GMM salvati in: {filename}")
    return filename

def plot_gmm_clusters(X, gmm, title="GMM Clustering Results"):
    """
    Visualizza i risultati del clustering
    """
    # Predici i cluster
    labels = gmm.predict(X)
    
    # Crea il plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    
    # Aggiungi i centroidi
    plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], color='red', marker='x', s=200, 
               linewidth=3, label='Centroidi')
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    # Genera dati di esempio
    X = create_sample_data(n_samples=20000000)
    
    # Salva il dataset
    data_filename = save_dataset(X, "../data/20M.csv")

    
    # Puoi anche ricaricare i dati così:
    # X = load_dataset(data_filename)

    # take time
    # start = datetime.datetime.now()
    # Addestra il modello GMM
    # gmm = fit_gmm(X)
    # end = datetime.datetime.now()
    # print(f"Tempo di addestramento: {end-start}")
    # print in second
    # Stampa i parametri
    #à print_gmm_parameters(gmm)

    # print(gmm.n_iter_)

    
    # Salva i parametri del modello
    # params_filename = save_gmm_parameters(gmm, "esempio_parametri_gmm")
    
    # Visualizza i risultati
    # plot_gmm_clusters(X, gmm)

if __name__ == "__main__":
    main()