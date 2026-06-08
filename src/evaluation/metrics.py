import numpy as np
import itertools
import logging
from sklearn.metrics import accuracy_score
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

def calculate_min_inter_cluster_distance(centers: np.ndarray) -> float:
    """
    Calculates the minimum Euclidean distance between any two cluster centers.
    Higher values indicate better separated clusters.
    """
    num_clusters = len(centers)
    min_dist = float('inf')
    
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < min_dist:
                min_dist = dist
                
    return min_dist if min_dist != float('inf') else 0.0

def calculate_inertia(data: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculates the total within-cluster sum of squared distances (Inertia).
    Lower values indicate tighter, more compact clusters.
    """
    inertia = 0.0
    for i, center in enumerate(centers):
        # Get all data points assigned to cluster i
        cluster_points = data[labels == i]
        if len(cluster_points) > 0:
            # Add squared distances from points to their center
            inertia += np.sum(np.linalg.norm(cluster_points - center, axis=1) ** 2)
            
    return inertia

def calculate_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray, num_clusters: int) -> float:
    """
    Calculates clustering accuracy by evaluating all possible permutations 
    of predicted labels to find the best match with ground truth labels.
    """
    # Generate all possible label mappings (0, 1, 2) -> (0, 1, 2), (0, 2, 1), etc.
    possible_labels = list(range(num_clusters))
    permutations = list(itertools.permutations(possible_labels))
    
    max_accuracy = 0.0
    
    for perm in permutations:
        # Create a mapping dictionary for this permutation
        mapping = {old_label: new_label for old_label, new_label in zip(possible_labels, perm)}
        
        # Apply the mapping to the predicted labels
        mapped_preds = np.array([mapping[label] for label in labels_pred])
        
        # Calculate accuracy for this permutation
        acc = accuracy_score(labels_true, mapped_preds) * 100.0
        
        if acc > max_accuracy:
            max_accuracy = acc
            
    return max_accuracy

def evaluate_clustering(data: np.ndarray, labels_true: np.ndarray, labels_pred: np.ndarray, centers: np.ndarray, num_clusters: int) -> Dict[str, float]:
    """
    A unified wrapper to execute all evaluation metrics at once.
    
    Returns:
        Dict[str, float]: A dictionary containing Accuracy, Min ICD, and Inertia.
    """
    logger.info("Evaluating clustering performance...")
    
    accuracy = calculate_accuracy(labels_true, labels_pred, num_clusters)
    min_icd = calculate_min_inter_cluster_distance(centers)
    inertia = calculate_inertia(data, centers, labels_pred)
    
    return {
        "accuracy": accuracy,
        "min_inter_cluster_distance": min_icd,
        "inertia": inertia
    }