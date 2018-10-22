nfrom copy import deepcopy
import numpy as np
from statistics import mean

from classification import get_stumps
from evaluation import central_test_accuracy, edges
from network import null_graph
from optimization import gd_reg_local_FW, kalo_graph_discovery, local_FW
from related_works import colearning
from utils import load_computer, get_split_per_list, get_min_max

# set graph of nodes with local personalized data

NB_ITER = 10000
random_state = 72018

CV_SPLITS = 3
MU_LIST = [10**i for i in range(-3, 3)]

STEP = 500

X, Y, X_test, Y_test, K, max_nb_instances = load_computer()
D = X[0].shape[1]

# get graph
vmin, vmax = get_min_max(X)
base_clfs = get_stumps(n=28, d=D, min_v=vmin, max_v=vmax)

nodes = null_graph(X, Y, X_test, Y_test, K, max_nb_instances)
local_FW(nodes, base_clfs, beta=10, nb_iter=NB_ITER, monitors={})

kalo_mu, kalo_b = 1, 1
init_w = kalo_graph_discovery(nodes, kalo_mu, kalo_b)
kalo = gd_reg_local_FW(nodes, base_clfs, init_w, gd_method={"name":"kalo", "pace_gd": STEP, "args":(kalo_mu, kalo_b)}, beta=10, mu=kalo_mu, nb_iter=NB_ITER, reset_step=False, monitors={})

results = {}.fromkeys(MU_LIST, 0.)

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=random_state):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    for mu in MU_LIST:

        print(mu)
        linear, _ = colearning(K, train_x, train_y, test_x, test_y, D, NB_ITER, kalo[-1]["adj-matrix"], kalo[-1]["similarities"], mu=mu, max_samples_per_node=max_nb_instances)

        results[mu] += linear[-1]["test-accuracy"]

print("best mu:", max(results, key=results.get))
