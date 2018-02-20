from itertools import combinations
import numpy as np
import math

from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_moons
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, scale, MinMaxScaler

def rotate(v1, v2):

    #rotate wrt s
    c = np.dot(v2, np.asarray([1,0])) / np.linalg.norm(v2)
    s = math.sin(math.acos(c))
    rotation = np.asarray([[c, -s], [s, c]])

    return np.dot(v1, rotation)

# ---------------------------------------------------------------------- LOAD DATASETS

def load_wine_dataset():
    
    X, Y = load_wine(return_X_y=True)

    # keep only two classes with labels -1,1
    indices = Y != 2
    Y, X = Y[indices], X[indices]
    Y[Y==0] = -1

    return scale(X), Y

def load_iris_dataset():
    
    X, Y = load_iris(return_X_y=True)

    # merge two classes, only two classes with labels -1,1
    Y[Y==0] = -1
    Y[Y==2] = 1

    return scale(X), Y

def load_breast_dataset():
    
    X, Y = load_breast_cancer(return_X_y=True)
    Y[Y==0] = -1

    return X, Y

def load_uci_dataset(name, y_pos=0):

    dataset = np.loadtxt(name)

    if y_pos == -1:
        x, y = np.split(dataset, [-1], axis=1)
    else:
        y, x = np.split(dataset, [1], axis=1)

    return scale(x), np.squeeze(y)

def load_csr_matrix(filename, y_pos=0):
    with open(filename,'r') as in_file:
        data, indices, indptr = [],[],[0]

        labels = []
        ptr = 0

        for line in in_file:
            line = line.split(None, 1)
            if len(line) == 1: 
                line += ['']
            label = line[y_pos]
            features = line[-1-y_pos]
            labels.append(float(label))

            f_list = features.split()
            for f in f_list:

                k,v = f.split(':')
                data.append(float(v))
                indices.append(float(k)-1)

            ptr += len(f_list)
            indptr.append(ptr)

        return csr_matrix((data, indices, indptr)), np.asarray(labels)

def load_sparse_dataset(name, y_pos=0):

    x, y = load_csr_matrix(name, y_pos)

    return scale(x, with_mean=False), y

def load_dense_dataset(name, y_pos=0):

    x, y = load_sparse_dataset(name, y_pos)

    return x.toarray(), y

# --------------------------------------------------------------- cross-validation

def get_split(x, nb_splits, shuffle=True, rnd_state=None):
    """ generator """
 
    splitter = KFold(n_splits=nb_splits)

    for train_index, test_index in splitter.split(x):

        yield train_index, test_index

def get_split_per_list(x, nb_splits, shuffle=True, rnd_state=None):
    """ generator """
    
    nb_lists = len(x)
    splitters = [KFold(n_splits=nb_splits) for _ in range(nb_lists)]

    for _ in range(nb_splits):

        yield [next(s.split(x[i])) for i, s in enumerate(splitters)]

# code adapted from https://gitlab.inria.fr/magnet/decentralizedpaper/blob/master/notebooks/basic_data_generation.ipynb
def generate_models(nb_clust=1, nodes_per_clust=100, inter_clust_stdev=0, intra_clust_stdev=np.sqrt(1/2), normalize_centroids=False, random_state=1):
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
            
    theta_true = np.vstack([rng.normal(loc=centroid, scale=intra_clust_stdev, size=(nodes_per_clust,2)) for centroid in centroids])
    n = len(theta_true)

    return n, theta_true, cluster_indexes

def generate_moons(n, theta_true, dim, min_samples_per_node=3, max_samples_per_node=20, samples_stdev=0.1, test_samples_per_node=100, sample_error_rate=5e-2, random_state=1):

    rng = np.random.RandomState(random_state)
    # scaler = MinMaxScaler((-1, 1))
    n_samples = rng.randint(min_samples_per_node, max_samples_per_node, size=n)
    c = n_samples / n_samples.max()
    C = np.diag(c)

    x, y = [], []
    for n_i, s in zip(n_samples, theta_true):
        x_i, y_i = make_moons(n_i, noise=samples_stdev, random_state=random_state)
        # x_i = scaler.fit_transform(x_i)
        x_i = np.hstack((rotate(x_i, s), np.zeros((n_i, dim - 2))))
        x.append(x_i)
        y_i[y_i==0] = -1
        y.append(y_i)

    x_test, y_test = [], []
    for s in theta_true:
        x_i, y_i = make_moons(test_samples_per_node, noise=samples_stdev, random_state=random_state)
        # x_i = scaler.fit_transform(x_i)
        x_i = np.hstack((rotate(x_i, s), np.zeros((test_samples_per_node, dim - 2))))
        x_test.append(x_i)
        y_i[y_i==0] = -1
        y_test.append(y_i)

    # Add noise
    for i in range(n):
        y[i][rng.choice(len(y[i]), replace=False, size=int(sample_error_rate*len(y[i])))] *= -1
        y_test[i][rng.choice(len(y_test[i]), replace=False, size=int(sample_error_rate*len(y_test[i])))] *= -1

    return n_samples, x, y, x_test, y_test, c, C

def generate_samples(n, theta_true, dim, min_samples_per_node=3, max_samples_per_node=20, samples_stdev=np.sqrt(1./2), test_samples_per_node=100, sample_error_rate=5e-2, random_state=1):
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

# ----------------------------------------------------------

def partition(x, y, nb_nodes, cluster_data=True, random_state=None):
    M, _ = x.shape
    
    if cluster_data:
        gm = GaussianMixture(nb_nodes, init_params="random", random_state=random_state)
        gm.fit(x)
        labels = gm.predict(x)
        groups = [[x[labels==i], y[labels==i]] for i in range(nb_nodes)]

    else:
        shuffled_ids = np.random.permutation(M)
        s = M // nb_nodes   
        groups = [[x[shuffled_ids][i*s:(i+1)*s], y[shuffled_ids][i*s:(i+1)*s]] for i in range(nb_nodes)]

    return groups

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

    return adjacency, similarities