import csv
from itertools import combinations
import numpy as np
import numpy.linalg as LA
from numpy.polynomial.polynomial import Polynomial, polyval2d
import math
import os
import random

from scipy.sparse import csr_matrix
from sklearn.datasets import load_breast_cancer, load_iris, load_wine, make_moons
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize, scale, MinMaxScaler

# ----------------------------------------------------------------------------- IO FUNCTIONS

def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def csv_to_dict(filename):

    with open(filename, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        my_dict = {row[0]: eval(row[1]) for row in reader}

    return my_dict

def dict_to_csv(my_dict, header, filename):

    make_directory(os.path.dirname(filename))

    with open(filename, 'w') as csv_file:
        writer = csv.writer(csv_file)
        
        writer.writerow(header)
        for key, value in my_dict.items():
            writer.writerow([key, value])

# --------------------------------------------------------------------------------- MONITORS

def stack_results(nodes, results, dual, monitors):
    """ modify results! """

    results.append({})  

    for k, call in monitors.items():
        results[-1][k] = call[0](nodes, *call[1])
    results[-1]["duality-gap"] = dual


# --------------------------------------------------------------------- ARRAY ROUTINES
def get_min_max(sample):
    
    vmin, vmax = float("inf"), -float("inf")

    for l in sample:
        vmin = min(vmin, np.min(l))
        vmax = max(vmax, np.max(l))

    return vmin, vmax

def square_root_matrix(matrix):
    """ square root of a symmetric PSD matrix.
    Due to numerical errors, the computed eigen values can be complex or negative. Here all imaginary parts are set to 0. """

    eig_values, eig_vectors = LA.eig(matrix)
    eig_values, eig_vectors = np.real(eig_values).clip(min=0), np.real(eig_vectors)

    sqrt_diag = np.diag(np.sqrt(eig_values))
    sqrt_matrix = np.dot(np.dot(eig_vectors, sqrt_diag), LA.pinv(eig_vectors))

    return sqrt_matrix

def rotate(v1, v2):

    #rotate wrt s
    c = np.dot(v2, np.asarray([1,0])) / np.linalg.norm(v2)

    s = math.sin(math.acos(c)*np.sign(v2[1]))

    rotation = np.asarray([[c, -s], [s, c]])

    return np.dot(v1, rotation)

# ---------------------------------------------------------------------- LOAD DATASETS

DATASET_PATH = os.path.join("datasets")

def load_school(path=DATASET_PATH, thr=20, split=1, moy=True):
    """ if moy is True, school attributes are averaged over the three years, otherwise
    139 x 3 tasks are created instead (a task = a school per year) """
    from scipy.io import loadmat

    DIR = os.path.join(path, "school_splits")
    STUDENT_FTS = np.r_[5:21,27]
    SCHOOL_YEAR_FTS = np.r_[:5,21:27]

    dataset = loadmat(os.path.join(DIR, "school_b.mat"))
    splits = loadmat(os.path.join(DIR, "school_{}_indexes.mat".format(split)))

    indexes, scores, features = dataset["task_indexes"].astype(int), dataset["y"].squeeze(), dataset["x"].astype(float).T

    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    # features = features.T

    tr_points, tr_indexes = splits["tr"].squeeze().astype(int), splits["tr_indexes"].squeeze().astype(int)
    ts_points, ts_indexes = splits["tst"].squeeze().astype(int), splits["tst_indexes"].squeeze().astype(int)

    train_fts = features[tr_points-1]
    train_sc = scores[tr_points-1]
    test_fts = features[ts_points-1]
    test_sc = scores[ts_points-1]

    x_train, y_train = [], []
    x_test, y_test = [], []

    end_train = tr_indexes[0] - 1
    end_test = ts_indexes[0] - 1

    max_nb_instances = 0

    tasks = []

    for i in range(1, 140):

        start_train = end_train
        start_test = end_test

        try:
            end_train = tr_indexes[i] - 1
            end_test = ts_indexes[i] - 1

        except:
            end_train = -1
            end_test = -1

        x_train.append(train_fts[start_train:end_train][:, STUDENT_FTS])
        y_train.append((train_sc[start_train:end_train] > thr) * 2 - 1) 

        x_test.append(test_fts[start_test:end_test][:, STUDENT_FTS])
        y_test.append((test_sc[start_test:end_test] > thr) * 2 - 1) 

        max_nb_instances = max(max_nb_instances, len(x_train[-1]))

        task_fts = np.unique(train_fts[start_train:end_train][:, SCHOOL_YEAR_FTS], axis=0)
        tasks.append(np.mean(task_fts[:, 3:], axis=0))

    task_similarities = cosine_similarity(tasks)
    task_similarities = sim_map(task_similarities, sigma=0.01)
    task_similarities[np.diag_indices(139)] = 0
    
    adjacency = get_adj_matrix(task_similarities)

    return x_train, y_train, x_test, y_test, adjacency, task_similarities, len(task_similarities), max_nb_instances

def load_computer(path=DATASET_PATH, thr=5, max_train_ratio=0.50, rnd_state=2018):

    from scipy.io import loadmat

    rng = np.random.RandomState(rnd_state)
    random.seed(rnd_state)

    dataset = loadmat(os.path.join(path, "conjointAnalysisComputerBuyers.mat"))

    scores, features = dataset["likeBuy"].astype(int), dataset["designMarix"].astype(int)

    max_nb_train = int(max_train_ratio * features.shape[0])
    max_nb_train = max(5, max_nb_train)
    computer_ids = rng.permutation(np.arange(0, 20))

    x_train, y_train = [], []
    x_test, y_test = [], []

    for i in range(190):

        nb_train = random.randint(5, max_nb_train)
        train_ids, test_ids = computer_ids[:nb_train], computer_ids[nb_train:]

        x_train.append(features[train_ids])
        x_test.append(features[test_ids])

        y_train.append((scores[i, train_ids] > thr) * 2 - 1)
        y_test.append((scores[i, test_ids] > thr) * 2 - 1)

    return x_train, y_train, x_test, y_test, 190, nb_train

def load_mobiact(path=DATASET_PATH):

    import pandas as pd

    USER_FILE = os.path.join(path, "Mobi_Users.csv")
    SETS_DIR = os.path.join(path, "Mobi_Generated") 
    NB_USERS = 67

    train_x, train_y, test_x, test_y = [], [], [], []
    users_not_found = []

    max_nb_instances = 1
    for user_id in range(NB_USERS):
        train_f = os.path.join(SETS_DIR, "user{}_train.csv".format(user_id+1))
        test_f = os.path.join(SETS_DIR, "user{}_test.csv".format(user_id+1))

        if os.path.isfile(train_f):
            train = pd.read_csv(train_f).as_matrix()[2:, 1:]
            test = pd.read_csv(test_f).as_matrix()[2:, 1:]

            train_labels = train[:, -1]
            train_labels[train_labels != "FOL"] = -1
            train_labels[train_labels == "FOL"] = 1

            test_labels = test[:, -1]
            test_labels[test_labels != "FOL"] = -1
            test_labels[test_labels == "FOL"] = 1

            train_x.append(np.array(train[:, :-1], dtype=np.float32))
            train_y.append(np.array(train_labels, dtype=np.int))
            test_x.append(np.array(test[:, :-1], dtype=np.float32))
            test_y.append(np.array(test_labels, dtype=np.int))

            max_nb_instances = max(max_nb_instances, len(train_x[-1]))

        else:
            users_not_found.append(user_id)

    users = pd.read_csv(USER_FILE, usecols=(3, 4, 5, 6))
    users.replace(('M', 'F', '-'), (1, 0, 0.5), inplace=True)
    users.drop(users.index[users_not_found], inplace=True)

    min_max_scaler = MinMaxScaler()
    scaled_users = min_max_scaler.fit_transform(users[['Age', 'Height', 'Weight', 'Gender']])
    user_distances = pairwise_distances(scaled_users)

    adjacency = get_adj_matrix(user_distances)

    return train_x, train_y, test_x, test_y, adjacency, user_distances, len(user_distances), max_nb_instances

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

# ----------------------------------------------------------------------- SYNTHETIC DATASETS

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
    max_nb_local_insts = n_samples.max()

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

    return n_samples, x, y, x_test, y_test, max_nb_local_insts

def generate_samples(n, theta_true, dim, min_samples_per_node=3, max_samples_per_node=20, samples_stdev=np.sqrt(1./2), test_samples_per_node=100, sample_error_rate=5e-2, random_state=1):
    """Generate train and test samples associated with nodes"""
    
    rng = np.random.RandomState(random_state)
    
    n_samples = rng.randint(min_samples_per_node, max_samples_per_node, size=n)
    max_nb_local_insts = n_samples.max()

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

    return n_samples, x, y, x_test, y_test, max_nb_local_insts

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

def get_adj_matrix(similarities, eps=1e-3):

    thresholds = similarities.max() * np.sqrt(10) ** (- np.arange(1, 100))
    for thresh in thresholds:
        adjacency = similarities > thresh
        if np.abs(np.linalg.eigvalsh(np.diag(adjacency.sum(axis=1)) - adjacency)[1]) > eps:
            break
    return adjacency

def compute_adjacencies(clfs, n, sigma=0.1):
    """Compute graph matrices according to true models of agents"""
    
    pairs = list(zip(*combinations(range(n),2)))
    similarities = np.zeros((n,n))
    norms_ = np.linalg.norm(clfs, axis=1)
    similarities[pairs] = similarities.T[pairs] = ((clfs[pairs[0],] * clfs[pairs[1],]).sum(axis=1)/ (norms_[pairs[0],] * norms_[pairs[1],]))
    
    similarities = sim_map(similarities, sigma)
    similarities[np.diag_indices(n)] = 0
    
    adjacency = get_adj_matrix(similarities)

    return adjacency, similarities