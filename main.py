import numpy as np
from sklearn.datasets import load_iris 
from normalization import z_score_normalization
from GSA import GSA
from FCM import fcm 
from groundTruth import get_ground_truth
from Evaluate import evaluate

def main():
    dataset_name = "iris"  # You can change this to "wine" or "glass"

    # Load and normalize data
    data = load_iris().data 
    normalized_data = z_score_normalization(data)

    # GSA Configuration
    num_clusters = 3
    max_iter = 50 
    G0 = 100
    alpha = 20
    beta = 1
    gamma = 0.95
    lb = normalized_data.min()
    ub = normalized_data.max()
    w1 = 0.6  
    w2 = 0.4  

    icds = []
    accuracies = []

    for _ in range(1):
        # Run GSA for initialization
        best_centers = GSA(normalized_data, max_iter, G0, alpha, beta, gamma, lb, ub, w1, w2)

        # Run FCM 
        membership_matrix, final_centers = fcm(normalized_data, best_centers)

        # icd, accuracy = evaluate(dataset_name, membership_matrix, normalized_data, final_centers)
        # icds.append(icd)
        # accuracies.append(accuracy)

        # print(f"Dataset: {dataset_name}")
        # print(f"Best ICD: {np.min(icds)}")
        # print(f"Worst ICD: {np.max(icds)}")
        # print(f"Average ICD: {np.mean(icds)} ± {np.std(icds)}")
        # print(f"Best Accuracy: {np.max(accuracies)}")
        # print(f"Worst Accuracy: {np.min(accuracies)}")
        # print(f"Average Accuracy: {np.mean(accuracies)} ± {np.std(accuracies)}")
if __name__ == "__main__":
    main()
