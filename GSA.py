import numpy as np
from sklearn.metrics import pairwise_distances
import random
from utils import calculate_icd , calculate_silhouette_coefficient

def assign_labels(data, centers):
    """
    Assign each data point to the cluster with the closest center.

    Parameters:
    - data: The normalized dataset (numpy array).
    - centers: The cluster centers.

    Returns:
    - labels: The cluster labels for each data point.
    """
    if centers.ndim == 1:
     centers = centers.reshape(1, -1)  # Reshape if necessary 

    labels = np.argmin(pairwise_distances(data, centers, metric='euclidean'), axis=1)

    return labels


def GSA(data, max_iter, G0, alpha, beta, gamma, lb, ub, w1, w2):
    """
    Apply the Gravitational Search Algorithm (GSA) to find the optimal cluster centers.

    Parameters:
    - data: The normalized dataset (numpy array).
    - max_iter: Maximum number of iterations.
    - G0: Initial gravitational constant.
    - alpha: Gravitational acceleration coefficient.
    - beta: Constant in the calculation of gravitational acceleration.
    - gamma: Damping ratio.
    - lb: Lower bound for the search space.
    - ub: Upper bound for the search space.
    - w1: Weight for ICD in fitness function.
    - w2: Weight for Silhouette Coefficient in fitness function.

    Returns:
    - best_centers: The best cluster centers found by GSA.
    """
    num_agents = len(data)
    num_clusters = len(data[0])
    agents = np.zeros((num_agents, num_clusters))
    velocities = np.zeros((num_agents, num_clusters))
    distances = np.zeros((num_agents, num_agents))

    # Initialization
    for i in range(num_agents):
        agents[i] = np.random.uniform(lb, ub, num_clusters)

    for t in range(max_iter):
        # Calculate distances
        for i in range(num_agents):
            for j in range(num_agents):
                distances[i, j] = np.linalg.norm(agents[i] - agents[j]) + 1e-8

        # Calculate mass
        mass = np.zeros(num_agents)
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    mass[i] += distances[i, j] ** (-beta)
        mass = mass / np.max(mass)

        # Calculate gravitational force
        acc = np.zeros((num_agents, num_clusters))
        for i in range(num_agents):
            for j in range(num_agents):
                if i != j:
                    acc[i] += alpha * random.random() * (mass[j] / (distances[i, j] ** beta)) * (agents[j] - agents[i])
        acc = acc + np.random.uniform(-1, 1, (num_agents, num_clusters))

        # Update velocity and position
        velocities = (gamma * velocities) + acc
        agents += velocities

        # Boundary control
        agents[agents < lb] = lb
        agents[agents > ub] = ub

    best_centers = agents[np.argmin([calculate_fitness(data, agent, w1, w2) for agent in agents])]
    return best_centers

def calculate_fitness(data, centers, w1, w2):
    """
    Calculate the fitness value for a set of cluster centers using the weighted sum of ICD and Silhouette Coefficient.

    Parameters:
    - data: The normalized dataset (numpy array).
    - centers: The cluster centers.
    - w1: Weight for ICD in fitness function.
    - w2: Weight for Silhouette Coefficient in fitness function.

    Returns:
    - fitness: The fitness value.
    """
    labels = assign_labels(data, centers)
    print(data.shape)
    print(centers.shape)

    icd = calculate_icd(data, labels , centers)
    silhouette_coefficient = calculate_silhouette_coefficient(data, labels)
    fitness = w1 * icd + w2 * silhouette_coefficient
    return fitness

