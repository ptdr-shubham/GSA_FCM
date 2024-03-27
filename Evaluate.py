import numpy as np
from utils import calculate_icd
from groundTruth import get_ground_truth
from sklearn.metrics import accuracy_score



def calculate_accuracy(labels_true, labels_pred):
    """
    Calculate the accuracy of the clustering result.

    Parameters:
    - labels_true: Ground truth labels for each data point.
    - labels_pred: Predicted cluster labels.

    Returns:
    - accuracy: The clustering accuracy.
    """
    return accuracy_score(labels_true, labels_pred)


def evaluate(dataset_name, membership_matrix, data, centers):
    """
    Evaluate the clustering performance using the Intra-Cluster Distance (ICD) and accuracy metrics.

    Parameters:
    - dataset_name: Name of the dataset.
    - membership_matrix: Membership matrix from the clustering algorithm.
    - data: The dataset (numpy array).
    - centers: Cluster centers.

    Returns:
    - icd: The Intra-Cluster Distance.
    - accuracy: The clustering accuracy.
    """
    # Load ground truth labels for the dataset
    ground_truth_labels = get_ground_truth(dataset_name)
    
    # Assign cluster labels based on the membership matrix
    cluster_labels = np.argmax(membership_matrix, axis=1)
    
    # Calculate Intra-Cluster Distance (ICD)
    icd = calculate_icd(data, cluster_labels, centers)
    
    # Calculate clustering accuracy
    accuracy = calculate_accuracy(ground_truth_labels, cluster_labels)
    
    return icd, accuracy



