from copy import deepcopy
import itertools
import numpy as np
from statistics import mean

from classification import get_stumps
from evaluation import central_test_accuracy
from network import null_graph
from optimization import gd_reg_local_FW, local_FW, kalo_graph_discovery
from utils import generate_fixed_moons, get_split_per_list, get_min_max

# set graph of nodes with local personalized data
NB_ITER = 10000
D = 20
B = 200
NOISE_R = 0.05
random_state = 2017
STEP = 500

CV_SPLITS = 3
MU_LIST = [10**i for i in range(-3, 3)]
B_LIST = [10**i for i in range(-3, 3)]
# MU_LIST = [1]
# BETA_LIST = [1]

K, X, Y, X_test, Y_test, max_nb_instances, theta_true, angles, groundtruth_adj_matrix = generate_fixed_moons(D, sample_error_rate=NOISE_R, rnd_state=random_state)

results = {}
results = results.fromkeys(itertools.product(MU_LIST, B_LIST), 0.)

vmin, vmax = get_min_max(X)
base_clfs = get_stumps(n=B, d=D, min_v=vmin, max_v=vmax)

local_nodes = null_graph(X, Y, X_test, Y_test, K, max_nb_instances)
local_FW(local_nodes, base_clfs, beta=10, nb_iter=NB_ITER, monitors={})

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=random_state):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    vmin, vmax = get_min_max(train_x)
    base_clfs = get_stumps(n=B, d=D, min_v=vmin, max_v=vmax)

    # set graph
    nodes = null_graph(train_x, train_y, test_x, test_y, K, max_nb_instances)

    for mu in MU_LIST:

        for b in B_LIST:

            print(mu, b)

            init_w = kalo_graph_discovery(local_nodes, mu, b)
            nodes_copy = deepcopy(nodes)
            gd_reg_local_FW(nodes_copy, base_clfs, gd_method={"name":"kalo", "pace_gd": STEP, "args":(mu, b)}, init_w=init_w, beta=10, mu=mu, nb_iter=NB_ITER, reset_step=False, monitors={})

            results[(mu, b)] += central_test_accuracy(nodes_copy)


print("best mu, b:", max(results, key=results.get))
