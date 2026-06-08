import numpy as np
import logging
from typing import Tuple, Optional
from src.config import config

logger = logging.getLogger(__name__)

class FuzzyCMeans:
    """
    Executes Fuzzy C-Means clusttering, optimally initialized with GSA centroids
    """

    def __init__(self, data:np.ndarray, num_clusters: int, init_centroids: Optional[np.ndarray] = None):
        self.data = data
        self.num_clusters = num_clusters
        self.n_samples, self.n_features = data.shape

        #Load hyperparameters from config
        self.m = config.fcm.m
        self.epsilon = config.fcm.epsilon
        self.max_iter = config.fcm.max_iter

        # Initialize centers (Use GSA centers if provided, otherwise random)
        if initial_centers is not None:
            self.centers = initial_centers.copy()
            logger.info("FCM initialized with provided centroids (e.g., from GSA).")
        else:
            random_indices = np.random.choice(self.n_samples, self.num_clusters, replace=False)
            self.centers = self.data[random_indices].copy()
            logger.info("FCM initialized with random centroids.")
            
        self.U = np.zeros((self.n_samples, self.num_clusters))

    def _update_membership(self) -> np.ndarray:
        """
        Updates the fuzzy membership matrix (U). Equation 4.9.
        """
        # Calculate distances from each point to each cluster center
        # Adding a tiny value (1e-8) to prevent division by zero
        distances = np.zeros((self.n_samples, self.num_clusters))
        for j in range(self.num_clusters):
            distances[:, j] = np.linalg.norm(self.data - self.centers[j], axis=1)
            
        distances = np.fmax(distances, 1e-8)
        
        # Calculate new membership matrix
        power = 2.0 / (self.m - 1)
        inv_distances = 1.0 / (distances ** power)
        new_U = inv_distances / np.sum(inv_distances, axis=1, keepdims=True)
        
        return new_U
    
    def _update_centers(self) -> np.ndarray:
        """
        Calculates new cluster centers based on membership weights. Equation 4.10.
        """
        um = self.U ** self.m
        new_centers = (um.T @ self.data) / np.sum(um.T, axis=1, keepdims=True)
        return new_centers

    def fit(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Executes the FCM iterative loop.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The final cluster centers and the membership matrix.
        """
        logger.info(f"Starting FCM optimization (max iterations: {self.max_iter})...")
        
        # Step 1: Initialize membership based on starting centers
        self.U = self._update_membership()
        
        for t in range(self.max_iter):
            U_old = self.U.copy()
            
            # Step 2: Update cluster centers
            self.centers = self._update_centers()
            
            # Step 3: Update membership matrix
            self.U = self._update_membership()
            
            # Step 4: Check for convergence
            max_diff = np.max(np.abs(self.U - U_old))
            if max_diff < self.epsilon:
                logger.info(f"FCM converged at iteration {t+1} with diff {max_diff:.8f}")
                break
                
        else:
            logger.warning(f"FCM reached max iterations ({self.max_iter}) without full convergence.")
            
        return self.centers, self.U