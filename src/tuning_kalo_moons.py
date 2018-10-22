from copy import deepcopy
import itertools
import numpy as np
from statistics import mean

from classification import get_stumps
from evaluation import central_test_accuracy
from network import null_graph
from optimization import gd_reg_local_FW
from utils import generate_models, generate_moons, get_split_per_list, get_min_max, kalo_utils

# set graph of nodes with local personalized data
NB_ITER = 3000 # 10000 for 100 nodes
K = 20 # 100 for 100 nodes
STEP = 300 # 500 for 100 nodes

D = 20
B = 200
NOISE_R = 0.05
random_state = 2018
BETA = 10

ITER = 3
MU_LIST = [10**i for i in range(-3, 3)]
LA_LIST = [10**i for i in range(-3, 3)]
# NU_LIST = [10**i for i in range(-3, 3)]

results = {}.fromkeys(itertools.product(MU_LIST, LA_LIST), 0.)

for i in range(2, ITER+2):
    _, theta_true, _ = generate_models(nb_clust=1, nodes_per_clust=K, random_state=random_state * i)
    _, train_x, train_y, test_x, test_y, max_nb_instances = generate_moons(K, theta_true, D, random_state=random_state * i, sample_error_rate=NOISE_R)

    vmin, vmax = get_min_max(train_x)
    base_clfs = get_stumps(n=B, d=D, min_v=vmin, max_v=vmax)

    # set graph
    nodes = null_graph(train_x, train_y, test_x, test_y, K, max_nb_instances)

    for mu in MU_LIST:

        for la in LA_LIST:

            print(mu, la)

            nodes_copy = deepcopy(nodes)
            gd_reg_local_FW(nodes_copy, base_clfs, gd_method={"name":"kalo", "pace_gd": STEP, "args":(mu, la)}, beta=BETA, mu=mu, nb_iter=NB_ITER, monitors={})

            results[(mu, la)] += central_test_accuracy(nodes_copy)

print("best mu, la:", max(results, key=results.get))
