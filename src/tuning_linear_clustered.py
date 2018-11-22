from copy import deepcopy
import itertools
import numpy as np
from statistics import mean

from network import synthetic_graph
from related_works import colearning, alternating_colearning
from utils import generate_fixed_moons

# set graph of nodes with local personalized data
NB_ITER = 8000
D = 20
B = 200
NOISE_R = 0.05
random_state = 112018

ITER = 3
BETA = 10
MU_LIST = [10**i for i in range(-3, 3)]
# MU_LIST = [1]

results = {}
results = results.fromkeys(MU_LIST, 0.)

for i in range(2, ITER+2):

    K, train_x, train_y, test_x, test_y, max_nb_instances, theta_true, _, _ = generate_fixed_moons(D, sample_error_rate=NOISE_R, rnd_state=random_state * i)

    _, adj_matrix, similarities = synthetic_graph(train_x, train_y, test_x, test_y, K, theta_true, max_nb_instances)

    for mu in MU_LIST:

        print(mu)
        linear, _ = colearning(K, train_x, train_y, test_x, test_y, D, NB_ITER, adj_matrix, similarities, mu, max_nb_instances, checkevery=NB_ITER-1)

        results[mu] += linear[-1]["test-accuracy"]

print(results)
print("best mu:", max(results, key=results.get))


# MU_LIST = [10**i for i in range(-3, 0)]
# LA_LIST = [10**i for i in range(0, 5)]

# STEP = 500
# # MU_LIST = [1]

# results = {}
# results = results.fromkeys(itertools.product(MU_LIST, LA_LIST), 0.)

# for i in range(2, ITER+2):

#     K, train_x, train_y, test_x, test_y, max_nb_instances, theta_true, _, _ = generate_fixed_moons(D, sample_error_rate=NOISE_R, rnd_state=random_state * i)

#     for mu in MU_LIST:

#         for la in LA_LIST:

#             print(mu, la)
#             linear, _ = alternating_colearning(K, train_x, train_y, test_x, test_y, D, NB_ITER, mu=mu, la=la, max_samples_per_node=max_nb_instances, pace_gd=STEP, checkevery=NB_ITER-1)

#             results[(mu, la)] += linear[-1]["test-accuracy"]

print("best mu, la:", max(results, key=results.get))
