import numpy as np

def get_ground_truth(dataset_name):
    """
    Get the ground truth labels for a given dataset.

    Parameters:
    - dataset_name: The name of the dataset (e.g., "iris").

    Returns:
    - ground_truth: The ground truth labels.
    """
    if dataset_name == "iris":
        from sklearn import datasets
        iris = datasets.load_iris()
        ground_truth = iris.target
    elif dataset_name == "wine":
        from sklearn import datasets
        wine = datasets.load_wine()
        ground_truth = wine.target
    elif dataset_name == "glass":
        from sklearn import datasets
        glass = datasets.load_glass()
        ground_truth = glass.target
    else:
        raise ValueError("Invalid dataset name.")
    
    return ground_truth
