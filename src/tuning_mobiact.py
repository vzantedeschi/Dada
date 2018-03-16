from copy import deepcopy
import itertools
import numpy as np
from statistics import mean

from classification import get_stumps
from evaluation import central_accuracy
from network import graph
from optimization import regularized_local_FW
from utils import load_mobiact, get_split_per_list, get_min_max

# set graph of nodes with local personalized data
NB_ITER = 50
MU_LIST = [10**i for i in range(-3, 4)]
BETA_LIST = [10**i for i in range(5)]
CV_SPLITS = 3

X, Y, X_test, Y_test, adjacency, similarities, nb_nodes = load_mobiact()

D = X[0].shape[1]
B = 5*D

vmin, vmax = get_min_max(X)
base_clfs = get_stumps(B, D+1, vmin, vmax)

# set callbacks for optimization analysis
callbacks = {
    'accuracy': [central_accuracy, []]
}

results = {}.fromkeys(itertools.product(MU_LIST, BETA_LIST), 0.)

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=None):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    # set graph
    nodes = graph(train_x, train_y, test_x, test_y, nb_nodes, adjacency, similarities)

    for mu in MU_LIST:

        for beta in BETA_LIST:

            print(mu, beta)
            nodes_copy = deepcopy(nodes)
            r = regularized_local_FW(nodes_copy, base_clfs, nb_iter=NB_ITER, beta=beta, mu=mu, callbacks=callbacks)

            # keep value of last iteration
            results[(mu, beta)] += r[NB_ITER]["accuracy"][1]

print("best mu, beta:", max(results, key=results.get))