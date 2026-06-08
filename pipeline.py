import logging
from src.config import config
from src.data_loader import load_dataset
from src.preprocessing import min_max_normalize
from src.models.gsa import GravitationalSearchAlgorithm
from src.models.fcm import FuzzyCMeans
from src.models.kmeans import KMeansBaseline
from src.evaluation.metrics import evaluate_clustering
from src.visualization.plots import plot_metric_comparison

# Set up logging for the pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_clustering_pipeline(dataset_name: str):
    """
    Executes the end-to-end clustering pipeline for a given dataset.
    """
    logger.info(f"=== Starting Pipeline for Dataset: {dataset_name.upper()} ===")
    
    # Step 1: Load and Preprocess Data
    X_raw, y_true = load_dataset(dataset_name)
    num_clusters = len(set(y_true))
    logger.info(f"Loaded {X_raw.shape[0]} samples with {X_raw.shape[1]} features. Target clusters: {num_clusters}")
    
    X_scaled = min_max_normalize(X_raw)
    logger.info("Data successfully normalized (Min-Max).")
    
    # Step 2: Run K-Means Baseline
    logger.info("--- Executing Baseline Model ---")
    kmeans = KMeansBaseline(data=X_scaled, num_clusters=num_clusters)
    kmeans_centers, kmeans_labels = kmeans.fit()
    kmeans_metrics = evaluate_clustering(X_scaled, y_true, kmeans_labels, kmeans_centers, num_clusters)
    
    # Step 3: Run GSA-FCM Hybrid
    logger.info("--- Executing GSA-FCM Hybrid Model ---")
    
    # Phase A: GSA Initialization
    gsa = GravitationalSearchAlgorithm(data=X_scaled, num_clusters=num_clusters, pop_size=10)
    optimized_initial_centers = gsa.optimize()
    
    # Phase B: FCM Clustering
    fcm = FuzzyCMeans(data=X_scaled, num_clusters=num_clusters, initial_centers=optimized_initial_centers)
    fcm_centers, fcm_membership = fcm.fit()
    
    # Convert fuzzy probabilities into hard labels for evaluation
    fcm_labels = fcm_membership.argmax(axis=1)
    hybrid_metrics = evaluate_clustering(X_scaled, y_true, fcm_labels, fcm_centers, num_clusters)
    
    # Step 4: Compare and Visualize Results
    logger.info("=== Final Results ===")
    logger.info(f"K-Means Metrics: {kmeans_metrics}")
    logger.info(f"GSA-FCM Metrics: {hybrid_metrics}")
    
    # Combine results for plotting
    final_results = {
        "Baseline (K-Means)": kmeans_metrics,
        "Hybrid (GSA-FCM)": hybrid_metrics
    }
    
    # Generate the comparison chart and display it
    plot_metric_comparison(final_results, dataset_name=dataset_name)
    
    logger.info("=== Pipeline Execution Complete ===")