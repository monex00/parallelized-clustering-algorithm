import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Legge i dati del modello
params = pd.read_csv("model_params.csv")
responsibilities = pd.read_csv("responsibilities.csv")

# Determina il numero di cluster e features
clusters = params['Cluster'].unique()
features = params['Feature'].unique()

# Legge i dati originali
data = pd.read_csv("../data/s3.csv", header=None)

# Calcola il cluster dominante per ogni dato
dominant_responsibilities = responsibilities.loc[responsibilities.groupby('DataPoint')['Responsibility'].idxmax()]
dominant_responsibilities = dominant_responsibilities.set_index('DataPoint')

# Allinea gli indici
dominant_responsibilities = dominant_responsibilities.reindex(data.index)

# Plot dei dati e dei cluster
colors = plt.colormaps['tab10'](np.linspace(0, 1, len(clusters)))

for cluster in clusters:
    dominant_mask = dominant_responsibilities['Cluster'] == cluster
    cluster_data = data[dominant_mask]
    plt.scatter(cluster_data[0], cluster_data[1], color=colors[int(cluster)], alpha=0.6, label=f"Cluster {cluster}")

# Plot delle mean
mean_matrix = params.pivot(index='Cluster', columns='Feature', values='Mean').values
for cluster in clusters:
    plt.scatter(mean_matrix[int(cluster), 0], mean_matrix[int(cluster), 1], c='black', marker='x', s=200, label=f"Cluster {cluster} Mean")

# plt.legend()
plt.title("Cluster Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("cluster_plot.png")
plt.show()
