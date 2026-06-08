import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Set global styling for publication-ready plots
sns.set_theme(style="whitegrid", palette="muted")

def plot_convergence(fitness_history: List[float], title: str = "GSA Fitness Convergence", save_path: str = None):
    """
    Plots the convergence curve of the Gravitational Search Algorithm over iterations.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(fitness_history, linewidth=2, color="#2ca02c", label="Best Fitness")
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Fitness Value", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Convergence plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()

def plot_metric_comparison(results: Dict[str, Dict[str, float]], dataset_name: str, save_path: str = None):
    """
    Creates a grouped bar chart comparing algorithms across multiple metrics.
    
    Args:
        results: Format expected -> {"K-Means": {"accuracy": 85, "inertia": 120}, "GSA-FCM": {"accuracy": 92, "inertia": 95}}
        dataset_name: Name of the dataset for the title.
    """
    algorithms = list(results.keys())
    # Extract the metric names from the first algorithm's results
    metrics = list(results[algorithms[0]].keys())
    
    # Set up the bar positions
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars for each algorithm
    for i, algo in enumerate(algorithms):
        scores = [results[algo][metric] for metric in metrics]
        offset = width * i - (width / 2 if len(algorithms) == 2 else 0)
        
        bars = ax.bar(x + offset, scores, width, label=algo)
        ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title(f'Performance Comparison on {dataset_name.capitalize()} Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    
    # Clean up metric names for the x-axis labels (e.g., "min_inter_cluster_distance" -> "Min Inter Cluster Distance")
    clean_metrics = [m.replace('_', ' ').title() for m in metrics]
    ax.set_xticklabels(clean_metrics, fontsize=11)
    
    ax.legend(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Comparison plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()