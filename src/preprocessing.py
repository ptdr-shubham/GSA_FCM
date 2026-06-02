import numpy as np

def min_max_normalize(data: np.ndarray) -> np.ndarray:
    """
    Scales the features of the dataset to the range [0, 1].
    
    Args:
        data (np.ndarray): The input numerical dataset to be normalized.
        
    Returns:
        np.ndarray: The Min-Max normalized dataset.
    """
    min_values = np.min(data,axis=0)
    max_values = np.max(data,axis=0)

    range_values = max_values - min_values
    # Avoid division by zero
    range_values[range_values == 0] = 1.0

    return (data  - min_values) / range_values


def z_score_normalize(data: np.ndarray) -> np.ndarray:
    """
    standardize features by removing the mean and scaaling to unit variance.
     
    Args:
        data (np.ndarray): The input numerical dataset to be standardized.
        
    Returns:
        np.ndarray: The Z-score standardized dataset.
    """

    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)

    #avoid division by zero
    std_dev[std_dev == 0] = 1.0

    return (data - mean) / std_dev