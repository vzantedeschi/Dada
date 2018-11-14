from copy import deepcopy

from classification import get_stumps
from evaluation import central_test_accuracy
from network import null_graph, get_alphas
from optimization import local_FW
from utils import generate_fixed_moons, get_split_per_list, get_min_max

NB_ITER = 1000
D = 20
B = 200
NOISE_R = 0.05
random_state = 112018

ITER = 3
BETA_LIST = [10**i for i in range(-3, 3)]

results = {}.fromkeys(BETA_LIST, 0.)

for i in range(2, ITER+2):

    K, train_x, train_y, test_x, test_y, max_nb_instances, theta_true, angles, groundtruth_adj_matrix = generate_fixed_moons(D, sample_error_rate=NOISE_R, rnd_state=random_state * i)

    vmin, vmax = get_min_max(train_x)
    base_clfs = get_stumps(n=B, d=D, min_v=vmin, max_v=vmax)

    # get nodes
    nodes = null_graph(train_x, train_y, test_x, test_y, K, max_nb_instances)

    for beta in BETA_LIST:

        print(beta)

        nodes_copy = deepcopy(nodes)
        local_FW(nodes_copy, base_clfs, beta=beta, nb_iter=NB_ITER, monitors={})

        results[beta] += central_test_accuracy(nodes_copy)

print(results)
print("best beta:", max(results, key=results.get))
