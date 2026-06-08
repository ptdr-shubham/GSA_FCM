import numpy as np
import logging
from typing import Tuple
from SRC.config import config


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')

class GravitationalSearchAlgorithm:
    """
    Optimizes initial cluster centroids using the laws of gravity and mass interactions.
    """

    def __init__(self, data: np.ndarray, num_clusters: int, pop_size: int = 10):
        self.data = data
        self.num_clusters = num_clusters
        self.pop_size = pop_size
        self.n_samples, self.n_features = data.shape

        # Load hyperparameters directly from centralized config
        self.G0 = config.gsa.G0
        self.alpha = config.gsa.alpha
        self.w1 = config.gsa.w1
        self.w2 = config.gsa.w2
        self.max_iter = config.gsa.max_iter

        # Initialize agents (potential cluster centers) and velocities
        self.bounds_min = np.min(data, axis=0)
        self.bounds_max = np.max(data, axis=0)

        self.agents = np.random.uniform(
            low = self.bounds_min,
            high = self.bounds_max,
            size=(self.pop_size, self.num_clusters, self.n_features)
        )
        self.velocities = np.zeros_like(self.agents)

    def _calculate_fitness(self, centers: np.ndarray) -> float:
        """
        Fitness is defined based on inter-cluster distance(maximize)
        and intra cluster variance(minimize)
        """
        # 1. Inter-cluster distance (a)
        min_inter_dist = float('inf')
        for i in range(self.num_clusters):
            for j in range(i + 1, self.num_clusters):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < min_inter_dist:
                    min_inter_dist = dist
        
        # Prevent infinity if clusters collapse
        if min_inter_dist == float('inf'):
            min_inter_dist = 0.0
            
        # 2. Intra-cluster variance (Inertia - b)
        distances = np.linalg.norm(self.data[:, np.newaxis] - centers, axis=2)
        min_distances = np.min(distances, axis=1)
        inertia = np.sum(min_distances ** 2)
        
        # Fitness = w1 * inter_distance - w2 * intra_variance
        return (self.w1 * min_inter_dist) - (self.w2 * inertia)

    def optimize(self) -> np.ndarray:
        """
        Executes the GSA optimization loop to find the best cluster centroids.
        
        Returns:
            np.ndarray: The optimized cluster centroids.
        """
        logger.info(f"Starting GSA optimization for {self.max_iter} iterations...")
        
        best_agent = None
        best_fitness = -float('inf')
        
        for t in range(self.max_iter):
            fitness_values = np.array([self._calculate_fitness(agent) for agent in self.agents])
            
            # Identify Best and Worst
            current_best_idx = np.argmax(fitness_values)
            current_worst_idx = np.argmin(fitness_values)
            
            if fitness_values[current_best_idx] > best_fitness:
                best_fitness = fitness_values[current_best_idx]
                best_agent = self.agents[current_best_idx].copy()
            
            # Calculate Gravitational Constant (G) - Equation 4.8
            G = self.G0 * np.exp(-self.alpha * (t / self.max_iter))
            
            # Calculate Masses - Equation 4.1 & 4.2
            worst_fit = fitness_values[current_worst_idx]
            best_fit = fitness_values[current_best_idx]
            
            # Prevent division by zero if all agents have the same fitness
            if best_fit == worst_fit:
                masses = np.ones(self.pop_size) / self.pop_size
            else:
                m_i = (fitness_values - worst_fit) / (best_fit - worst_fit)
                masses = m_i / np.sum(m_i)
                
            # Calculate Forces, Accelerations, and Update Positions
            epsilon = 1e-7
            for i in range(self.pop_size):
                force = np.zeros((self.num_clusters, self.n_features))
                for j in range(self.pop_size):
                    if i != j:
                        # Euclidean distance between agent i and agent j
                        distance = np.linalg.norm(self.agents[i] - self.agents[j])
                        # Equation 4.3
                        force += np.random.rand() * masses[j] * (self.agents[j] - self.agents[i]) / (distance + epsilon)
                
                # Equation 4.4, 4.5, 4.6
                acceleration = force * G
                self.velocities[i] = (np.random.rand() * self.velocities[i]) + acceleration
                self.agents[i] = self.agents[i] + self.velocities[i]
                
                # Boundary Control (Bounding)
                self.agents[i] = np.clip(self.agents[i], self.bounds_min, self.bounds_max)
                
        logger.info("GSA optimization complete.")
        return best_agent