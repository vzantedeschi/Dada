# imports
from copy import deepcopy
from math import log

from classification import get_stumps
from network import exponential_graph
from optimization import block_kalo_graph_discovery, local_FW
from utils import generate_fixed_moons, get_min_max

def expected_new_nodes(nb_nodes, kappa, nb_iter):
    return round(kappa * ((nb_nodes**2 - nb_nodes - kappa) / nb_nodes * (nb_nodes))**nb_iter, 2)

# set graph of nodes with local personalized data
D = 20
n = 200
NOISE_R = 0.05
random_state = 2017

BETA = 10

K, X, Y, X_test, Y_test, max_nb_instances, theta_true, angles, groundtruth_adj_matrix = generate_fixed_moons(D, sample_error_rate=NOISE_R, rnd_state=random_state)

# set graph
nodes, _, _ = exponential_graph(X, Y, X_test, Y_test, K, theta_true, max_nb_instances)

# get weak classifiers
vmin, vmax = get_min_max(X)
base_clfs = get_stumps(n=n, d=D+1, min_v=vmin, max_v=vmax)

local_nodes = deepcopy(nodes)
local_FW(local_nodes, base_clfs, beta=BETA, nb_iter=1000, monitors={})

from optimization import block_kalo_graph_discovery
from utils import kalo_utils

S, triu_ix, map_idx = kalo_utils(K)
MU, LA = 0.1, 1

NB_ITER = 5
results_dict = {}

# starting from local models

kappas = [30, 50, 99]
for kappa in kappas:
    results_dict[kappa] = []
    
    for _ in range(NB_ITER):
        _, obj = block_kalo_graph_discovery(local_nodes, None, S, triu_ix, map_idx, mu=MU, la=LA, kappa=kappa, max_iter=3e5, monitor=True)
        results_dict[kappa].append(obj)

kappas = [1, 5, 10]
for kappa in kappas:
    results_dict[kappa] = []
    
    for _ in range(NB_ITER):
        _, obj = block_kalo_graph_discovery(local_nodes, None, S, triu_ix, map_idx, mu=MU, la=LA, kappa=kappa, max_iter=1e6, monitor=True)
        results_dict[kappa].append(obj)


# average NB_ITER iterations
kappas = sorted(results_dict.keys())

objectives = {}
    
for kappa in kappas:
    avgs = [sum(e) / NB_ITER for e in zip(*results_dict[kappa])]
    objectives[kappa] = avgs

for kappa in [30, 50, 99]:
    objectives[kappa] += [objectives[kappa][-1]] * int(1e6 - 3e5)

# communication

cost = min(2000 / K, n)
Z = 32

comm_iter = {}
comm = {}

for kappa in kappas:
    comm_iter[kappa] = [2 * Z * kappa + (cost * (Z + log(n)) + Z) * expected_new_nodes(K, kappa, i) for i in range(len(objectives[kappa]))]

for kappa in kappas:
    for i, _ in enumerate(comm_iter[kappa]):
        comm[kappa][i] = sum(comm_iter[kappa][:i])

max_comm = max([comm[kappa][-1] for kappa in kappas])

for kappa in kappas:
    comm[kappa].append(max_comm)
    objectives[kappa].append(objectives[kappa][-1])

# convergence
plt.figure(1, figsize=(12, 5))

plt.subplot(121)
plt.xlabel('nb iterations', fontsize=15)
plt.ylabel('obj function', fontsize=15)

cmap = plt.get_cmap("viridis")

for kappa in kappas:

    plt.plot([i*K for i in range(len(objectives[kappa]))], objectives[kappa], color=cmap(log(kappa) / log(99)), label="kappa = {}".format(kappa), linewidth=2)

plt.legend(loc='upper right', fontsize=12)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.xlim(0, 1e6)

plt.subplot(122)
plt.xlabel('communication', fontsize=15)
plt.ylabel('obj function', fontsize=15)

for kappa in kappas:
    plt.plot(comm[kappa], objectives[kappa], color=cmap(log(kappa) / log(99)), label="kappa = {}".format(kappa), linewidth=2)

plt.plot([full_gd_comm], [min_obj], color="o", label="centralized", linewidth=2)
    
plt.legend(loc='upper right', fontsize=12)

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12)

plt.xlim(0, max_comm)
plt.savefig("kappas.pdf",  bbox_inches="tight")