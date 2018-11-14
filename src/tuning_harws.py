from copy import deepcopy
import itertools

from classification import get_stumps
from evaluation import central_test_accuracy
from network import null_graph
from optimization import gd_reg_local_FW
from utils import load_harws, get_split_per_list, get_min_max

NB_ITER = 500
B = 1122
BETA = 1
random_state = 72018

CV_SPLITS = 2
MU_LIST = [10**i for i in range(-3, 3)]
LA_LIST = [10**i for i in range(-3, 3)]

# MU_LIST = [1]
# LA_LIST = [1]
STEP = 50

X, Y, _, _, N, max_nb_instances = load_harws(walking=True)
D = X[0].shape[1]

results = {}.fromkeys(itertools.product(MU_LIST, LA_LIST), 0.)

for indices in get_split_per_list(X, CV_SPLITS, rnd_state=random_state):

    train_x, test_x, train_y, test_y = [], [], [], []

    for i, inds in enumerate(indices):
        train_x.append(X[i][inds[0]])
        test_x.append(X[i][inds[1]])
        train_y.append(Y[i][inds[0]])
        test_y.append(Y[i][inds[1]])

    vmin, vmax = get_min_max(train_x)
    base_clfs = get_stumps(n=B, d=D, min_v=vmin, max_v=vmax)

    # get nodes
    nodes = null_graph(train_x, train_y, test_x, test_y, N, max_nb_instances)

    for mu in MU_LIST:

        for la in LA_LIST:

            print(mu, la)

            nodes_copy = deepcopy(nodes)
            gd_reg_local_FW(nodes_copy, base_clfs, gd_method={"name":"kalo", "pace_gd": STEP, "args":(mu, la)}, beta=BETA, mu=mu, nb_iter=NB_ITER, monitors={})

            results[(mu, la)] += central_test_accuracy(nodes_copy)

print(results)
print("best mu, la:", max(results, key=results.get))
