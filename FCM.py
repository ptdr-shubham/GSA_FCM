import numpy as np
from utils import calculate_icd, calculate_silhouette_coefficient

def fcm(data, cluster_centers, fuzziness_parameter=2, max_iterations=100, tolerance=1e-4):
    """
    Fuzzy C-Means clustering algorithm.

    Parameters:
    - data: The dataset (numpy array).
    - cluster_centers: Initial cluster centers.
    - fuzziness_parameter: Fuzziness parameter (default=2).
    - max_iterations: Maximum number of iterations (default=100).
    - tolerance: Convergence tolerance (default=1e-4).

    Returns:
    - membership_matrix: Final membership matrix.
    - cluster_centers: Final cluster centers.
    """
    n_data_points, n_features = data.shape
    n_clusters = len(cluster_centers)

    # Initialize membership matrix randomly
    membership_matrix = np.random.rand(n_data_points, n_clusters)
    membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1, keepdims=True)

    iteration = 0
    while iteration < max_iterations:
        # Update cluster centers
        numerator = np.dot(data.T, membership_matrix ** fuzziness_parameter)
        denominator = np.sum(membership_matrix ** fuzziness_parameter, axis=0)
        new_cluster_centers = numerator / denominator

        # Update membership matrix
        distances = np.linalg.norm(data[:, None] - new_cluster_centers, axis=2)
        new_membership_matrix = 1 / np.sum((distances[:, :, None] / distances[:, :, None]) ** (2 / (fuzziness_parameter - 1)), axis=1)

        # Check for convergence
        if np.linalg.norm(new_cluster_centers - cluster_centers) < tolerance:
            break

        cluster_centers = new_cluster_centers
        membership_matrix = new_membership_matrix

        iteration += 1

    return membership_matrix, cluster_centers

# # Example usage
# if __name__ == "__main__":
#     # Assuming 'data' is the normalized dataset and 'initial_cluster_centers' are the initial cluster centers from GSA
#     membership_matrix, cluster_centers = fcm(data, initial_cluster_centers)
#     print("Membership Matrix:\n", membership_matrix)
#     print("Cluster Centers:\n", cluster_centers)
