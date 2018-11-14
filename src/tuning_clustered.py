from copy import deepcopy
import itertools
import numpy as np
from statistics import mean

from classification import get_stumps
from evaluation import central_test_accuracy
from network import synthetic_graph, null_graph
from optimization import regularized_local_FW, gd_reg_local_FW
from utils import generate_fixed_moons, get_split_per_list, get_min_max

# set graph of nodes with local personalized data
NB_ITER = 8000
D = 20
B = 200
NOISE_R = 0.05
random_state = 112018

ITER = 3
BETA = 10


# MU_LIST = [10**i for i in range(-3, 3)]
# # MU_LIST = [1]

# results = {}
# results = results.fromkeys(MU_LIST, 0.)

# for i in range(2, ITER+2):

#     K, train_x, train_y, test_x, test_y, max_nb_instances, theta_true, angles, groundtruth_adj_matrix = generate_fixed_moons(D, sample_error_rate=NOISE_R, rnd_state=random_state * i)

#     vmin, vmax = get_min_max(train_x)
#     base_clfs = get_stumps(n=B, d=D, min_v=vmin, max_v=vmax)

#     # set graph
#     nodes, _, _ = synthetic_graph(train_x, train_y, test_x, test_y, K, theta_true, max_nb_instances)

#     for mu in MU_LIST:

#         print(mu)
#         nodes_copy = deepcopy(nodes)
#         regularized_local_FW(nodes_copy, base_clfs, nb_iter=NB_ITER, beta=BETA, mu=mu, monitors={})

#         results[mu] += central_test_accuracy(nodes_copy)

# print(results)
# print("best mu:", max(results, key=results.get))

MU_LIST = [10**i for i in range(-3, 3)]
LA_LIST = [10**i for i in range(-3, 3)]


STEP = 500
# MU_LIST = [1]

results = {}
results = results.fromkeys(itertools.product(MU_LIST, LA_LIST), 0.)

for i in range(2, ITER+2):

    K, train_x, train_y, test_x, test_y, max_nb_instances, theta_true, angles, groundtruth_adj_matrix = generate_fixed_moons(D, sample_error_rate=NOISE_R, rnd_state=random_state * i)

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
