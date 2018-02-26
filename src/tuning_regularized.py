from copy import deepcopy
import numpy as np
from statistics import mean

from evaluation import central_accuracy
from network import synthetic_graph
from optimization import regularized_local_FW
from utils import generate_models, generate_samples, get_split_per_list

# set graph of nodes with local personalized data
NB_ITER = 500
N = 100
D = 20
NOISE_R = 0.05
random_state = 2017

CV_SPLITS = 3
MU_LIST = [10**i for i in range(-3, 4)]

V, theta_true, cluster_indexes = generate_models(nb_clust=1, nodes_per_clust=N, random_state=random_state)
_, X, Y, _, _, _, _ = generate_samples(V, theta_true, D, random_state=random_state, sample_error_rate=NOISE_R)

# set callbacks for optimization analysis
callbacks = {
    'accuracy': [central_accuracy, []]
}

base_clfs_args = {
    'n': 2*D,
    'd': D+1,
}

results = {mu: 0. for mu in MU_LIST}

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=random_state):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    # set graph
    nodes, _, _ = synthetic_graph(train_x, train_y, test_x, test_y, V, theta_true)

    for mu in MU_LIST:

        nodes_copy = deepcopy(nodes)
        r = regularized_local_FW(nodes_copy, clfs_g_args=base_clfs_args, nb_iter=NB_ITER, mu=mu, callbacks=callbacks)

        # keep value of last iteration
        results[mu] += r[NB_ITER]["accuracy"][1]

print("best mu", max(results, key=results.get))