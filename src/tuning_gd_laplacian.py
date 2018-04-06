from copy import deepcopy
import itertools
import numpy as np
from statistics import mean

from classification import get_stumps
from evaluation import central_accuracy
from network import synthetic_graph
from optimization import gd_reg_local_FW
from utils import generate_models, generate_moons, get_split_per_list, get_min_max

# set graph of nodes with local personalized data
NB_ITER = 200
N = 20
D = 20
B = 200
NOISE_R = 0.05
random_state = 2017

CV_SPLITS = 3
MU_LIST = [10**i for i in range(-3, 4)]
BETA_LIST = [10**i for i in range(5)]
STEP_LIST = list(range(10, 80, 10))

V, theta_true, cluster_indexes = generate_models(nb_clust=1, nodes_per_clust=N, random_state=random_state)
_, X, Y, _, _, max_nb_instances = generate_moons(V, theta_true, D, random_state=random_state, sample_error_rate=NOISE_R)

results = {}
results = results.fromkeys(itertools.product(MU_LIST, BETA_LIST), 0.)

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=random_state):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    vmin, vmax = get_min_max(train_x)
    base_clfs = get_stumps(n=B, d=D+1, min_v=vmin, max_v=vmax)

    # set graph
    nodes, _, _ = synthetic_graph(train_x, train_y, test_x, test_y, V, theta_true, max_nb_instances)

    for mu in MU_LIST:

        for beta in BETA_LIST:

            for step in STEP_LIST:


                print(mu, beta, step)

                nodes_copy = deepcopy(nodes)
                gd_reg_local_FW(nodes_copy, base_clfs, gd_method={"name":"laplacian", "pace_gd": step, "args":(1)}, beta=beta, mu=mu, nb_iter=NB_ITER, reset_step=False, callbacks={})

                results[(mu, beta)] += central_accuracy(nodes_copy)[1]


print("best mu, beta:", max(results, key=results.get))
