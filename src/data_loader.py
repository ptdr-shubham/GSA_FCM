import numpy as np
from sklearn import datasets
from typing import Tuple

def load_dataset(dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads benchmark datasets from scikit-learn.
    
    Args:
        dataset_name (str): The name of the dataset ('iris', 'wine', or 'cancer').
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix (X) 
                                       and the target labels (y).
                                       
    Raises:
        ValueError: If an unsupported dataset name is provided.
    """
    if dataset_name == "iris":
        data = datasets.load_iris()
    elif dataset_name == "wine":
        data = datasets.load_wine()
    elif dataset_name == "cancer":
        data = datasets.load_breast_cancer()
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Supported datasets are: 'iris', 'wine', 'cancer'.")
    
    #Return features (X) and ground truth labels (y)
    return data.data, data.target 