from itertools import combinations
import numpy as np

from sklearn.datasets import load_wine
from sklearn.preprocessing import normalize, scale

# ---------------------------------------------------------------------------------- LOAD DATASETS

def load_wine_dataset():
    
    X, Y = load_wine(return_X_y=True)

    # keep only two classes with labels -1,1
    indices = Y != 2
    Y, X = Y[indices], X[indices]
    Y[Y==0] = -1

    return scale(X), Y

# code adapted from https://gitlab.inria.fr/magnet/decentralizedpaper/blob/master/notebooks/basic_data_generation.ipynb
def generate_models(nb_clust=1, inter_clust_stdev=0, intra_clust_stdev=np.sqrt(1/2),
                    nodes_per_clust=100, normalize_centroids=False, random_state=1):
    """Generate true models of all the agents"""

    rng = np.random.RandomState(random_state)
    
    if inter_clust_stdev > 0:
        centroids = rng.normal(size=(nb_clust, 2), scale=inter_clust_stdev)
        if normalize_centroids:
            centroids /= np.linalg.norm(centroids, axis=1)[:,None]
    else:
        centroids = np.zeros((nb_clust, 2))
    
    cluster_indexes = []
    start = 0
    for _ in range(nb_clust):
        cluster_indexes.append(np.arange(start, start+nodes_per_clust))
        start += nodes_per_clust
            
    theta_true = np.vstack([rng.normal(loc=centroid,
                                              scale=intra_clust_stdev,
                                              size=(nodes_per_clust,2))
                                              for centroid in centroids])
    n = len(theta_true)

    return n, theta_true, cluster_indexes

def generate_samples(n, theta_true, dim, min_samples_per_node=1, max_samples_per_node=20,
                     samples_stdev=np.sqrt(1./2), test_samples_per_node=100, sample_error_rate=5e-2, random_state=1):
    """Generate train and test samples associated with nodes"""
    
    rng = np.random.RandomState(random_state)
    
    n_samples = rng.randint(min_samples_per_node, max_samples_per_node, size=n)
    c = n_samples / n_samples.max()
    C = np.diag(c)

    x, y = [], []
    for n_i, s in zip(n_samples, theta_true):
        x_i = rng.normal(size=(n_i, dim), scale=samples_stdev)
        x.append(x_i)
        y.append((x_i[:, :2].dot(s) > 0)*2 - 1)

    x_test, y_test = [], []
    for s in theta_true:
        x_i = rng.normal(size=(test_samples_per_node, dim), scale=samples_stdev)
        x_test.append(x_i)
        y_test.append((x_i[:, :2].dot(s) > 0)*2 - 1)
    
    # Add noise
    for i in range(n):
        y[i][rng.choice(len(y[i]), replace=False, size=int(sample_error_rate*len(y[i])))] *= -1
        y_test[i][rng.choice(len(y_test[i]), replace=False, size=int(sample_error_rate*len(y_test[i])))] *= -1

    return n_samples, x, y, x_test, y_test, c, C

def sim_map(arr, sigma):
    """Function used to map [-1,1] into [0,2]
    (length of arc between two points on the unit circle)"""
    return np.exp(-((1-arr)**2 + (1-arr**2)) / (2*sigma))

def compute_adjacencies(theta_true, n, sigma=0.1):
    """Compute graph matrices according to true models of agents"""
    
    pairs = list(zip(*combinations(range(n),2)))
    similarities = np.zeros((n,n))
    norms_ = np.linalg.norm(theta_true, axis=1)
    similarities[pairs] = similarities.T[pairs] = ((theta_true[pairs[0],] * theta_true[pairs[1],]).sum(axis=1)/ (norms_[pairs[0],] * norms_[pairs[1],]))
    
    similarities = sim_map(similarities, sigma)
    similarities[np.diag_indices(n)] = 0
    
    thresholds = similarities.max() * np.sqrt(10) ** (- np.arange(1, 100))
    for thresh in thresholds:
        adjacency = similarities > thresh
        if np.abs(np.linalg.eigvalsh(np.diag(adjacency.sum(axis=1)) - adjacency)[1]) > 1e-3:
            break

    return adjacency