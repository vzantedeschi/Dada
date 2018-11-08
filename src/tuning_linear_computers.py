import itertools

import numpy as np

from related_works import alternating_colearning
from utils import load_computer, get_split_per_list

# set graph of nodes with local personalized data

NB_ITER = 1000
random_state = 72018

CV_SPLITS = 3
MU_LIST = [10**i for i in range(-3, 3)]
LA_LIST = [10**i for i in range(-3, 3)]

STEP = 10

X, Y, X_test, Y_test, K, max_nb_instances = load_computer()
D = X[0].shape[1]

results = {}.fromkeys(itertools.product(MU_LIST, LA_LIST), 0.)

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=random_state):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    for mu in MU_LIST:

        for la in LA_LIST:

            print(mu, la)
            linear, _ = alternating_colearning(K, train_x, train_y, test_x, test_y, D, NB_ITER, mu=mu, la=la, max_samples_per_node=max_nb_instances)

            results[(mu, la)] += linear[-1]["test-accuracy"]

print("best mu, la:", max(results, key=results.get))
