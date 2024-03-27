import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.metrics import  silhouette_samples

def calculate_icd(data, labels, centers):
    """
    Calculate the Intra-Cluster Distance (ICD) for a clustering result.

    Parameters:
    - data: The dataset (numpy array).
    - labels: Cluster labels for each data point.
    - centers: Cluster centers.

    Returns:
    - icd: The Intra-Cluster Distance.
    """
    k = len(centers)
    icd = 0
    for i in range(k):
        cluster_points = data[labels == i]
        center = centers[i].reshape(1, -1)  # Ensure center is 2D
        icd += np.sum(pairwise_distances(cluster_points, center, metric='euclidean'))
    return icd


def calculate_silhouette_coefficient(data, labels):
    """
    Calculate the Silhouette Coefficient for a clustering result.

    Parameters:
    - data: The dataset (numpy array).
    - labels: Cluster labels for each data point.

    Returns:
    - silhouette_coefficient: The Silhouette Coefficient.
    """
    return np.mean(silhouette_samples(data, labels))
