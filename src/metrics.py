import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness

def calculate_preservation_at_k(X, X_reduced, k=5):
    n_samples = X.shape[0]
    X_flat = X.reshape(n_samples, -1)
    X_red_flat = X_reduced.reshape(n_samples, -1)
    
    nn_orig = NearestNeighbors(n_neighbors=k+1).fit(X_flat)
    _, idx_orig = nn_orig.kneighbors(X_flat)
    
    nn_red = NearestNeighbors(n_neighbors=k+1).fit(X_red_flat)
    _, idx_red = nn_red.kneighbors(X_red_flat)
    
    scores = []
    for i in range(n_samples):
        set_orig = set(idx_orig[i, 1:k+1])
        set_red = set(idx_red[i, 1:k+1])
        scores.append(len(set_orig.intersection(set_red)) / k)
    return np.mean(scores)

def calculate_trustworthiness(X, X_reduced, k=5):
    X_flat = X.reshape(X.shape[0], -1)
    X_red_flat = X_reduced.reshape(X_reduced.shape[0], -1)
    return trustworthiness(X_flat, X_red_flat, n_neighbors=min(k, X.shape[0]-1))