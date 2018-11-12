import itertools
import numpy as np

from network import synthetic_graph
from related_works import alternating_colearning, colearning
from utils import generate_models, generate_moons

# set graph of nodes with local personalized data
NB_ITER = 500
K = 100
STEP = 50

D = 20
NOISE_R = 0.05
random_state = 2018

ITER = 3
MU_LIST = [10**i for i in range(-3, 3)]
LA_LIST = [10**i for i in range(-3, 3)]
# MU_LIST = [10**i for i in range(1)]
# LA_LIST = [10**i for i in range(1)]

results = {}.fromkeys(itertools.product(MU_LIST, LA_LIST), 0.)

for i in range(2, ITER+2):
    _, theta_true, _ = generate_models(nb_clust=1, nodes_per_clust=K, random_state=random_state * i)
    _, train_x, train_y, test_x, test_y, max_nb_instances = generate_moons(K, theta_true, D, random_state=random_state * i, sample_error_rate=NOISE_R)

    _, adj_matrix, similarities = synthetic_graph(train_x, train_y, test_x, test_y, K, theta_true, max_nb_instances)

    for mu in MU_LIST:

        # #uncomment for learning graph
        for la in LA_LIST:

            print(mu, la)

            linear, _ = alternating_colearning(K, train_x, train_y, test_x, test_y, D, NB_ITER, mu, la, max_nb_instances, pace_gd=STEP, checkevery=NB_ITER-1)

            results[(mu, la)] += linear[-1]["test-accuracy"]

        # #uncomment for learning only models
        # la = 1
        # linear, _ = colearning(K, train_x, train_y, test_x, test_y, D, NB_ITER, adj_matrix, similarities, mu, max_nb_instances, checkevery=NB_ITER-1)
        

        # results[(mu, la)] += linear[-1]["test-accuracy"]

print("best mu, la:", max(results, key=results.get))
