import numpy as np
import logging
from sklearn.cluster import KMeans
from typing import Tuple
from src.config import config

logger = logging.getLogger(__name__)

class KMeansBaseline:
    """
    A wrapper for scikit-learn's K-Means to serve as a performance baseline.
    """
    def __init__(self, data: np.ndarray, num_clusters: int):
        self.data = data
        self.num_clusters = num_clusters
        self.random_seed = config.random_seed
        
        # Initialize the scikit-learn model
        self.model = KMeans(
            n_clusters=self.num_clusters, 
            random_state=self.random_seed,
            n_init=10 # Explicitly set to avoid scikit-learn warnings
        )

    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the standard K-Means clustering algorithm.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The final cluster centers and the predicted labels.
        """
        logger.info("Starting standard K-Means baseline clustering...")
        
        # Fit the model and extract predictions
        predicted_labels = self.model.fit_predict(self.data)
        centers = self.model.cluster_centers_
        
        logger.info("K-Means clustering complete.")
        
        return centers, predicted_labels