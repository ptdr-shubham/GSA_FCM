import numpy as np

def z_score_normalization(data):
    """
    Perform Z-score normalization on the input data.

    Parameters:
    - data: The input dataset (numpy array).

    Returns:
    - normalized_data: The Z-score normalized dataset.
    """
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    normalized_data = (data - mean) / std_dev
    return normalized_data