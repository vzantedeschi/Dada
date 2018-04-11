from copy import deepcopy
import numpy as np
from statistics import mean

from classification import get_basis
from evaluation import central_accuracy, central_loss, accuracies, edges
from network import line_network, synthetic_graph, true_theta_graph
from optimization import centralized_FW, regularized_local_FW, local_FW, async_regularized_local_FW, global_regularized_local_FW, gd_reg_local_FW
from related_works import lafond_FW, colearning
from utils import generate_models, generate_samples

import matplotlib.pyplot as plt

# set graph of nodes with local personalized data
NB_ITER = 100
N = 20
D = 20
NOISE_R = 0.05
random_state = 2017
BETA = 1 # if None, simplex constraint
MU = 1

V, theta_true, cluster_indexes = generate_models(nb_clust=1, nodes_per_clust=N, random_state=random_state)
_, X, Y, X_test, Y_test, max_nb_instances = generate_samples(V, theta_true, D, random_state=random_state, sample_error_rate=NOISE_R)

# set graph
nodes, adj_matrix, similarities = synthetic_graph(X, Y, X_test, Y_test, V, theta_true, max_nb_instances)

# set callbacks for optimization analysis
callbacks = {
    'accuracy': [central_accuracy, []],
    'loss': [central_loss, []],
    'edges': [edges, []]
}

base_clfs = get_basis(n=D+1, d=D+1)

results = {}
hist_accuracies = {}

nodes_centralized = deepcopy(nodes)
results["centralized"] = centralized_FW(nodes_centralized, base_clfs, nb_iter=NB_ITER, beta=BETA, callbacks=callbacks)
hist_accuracies["centralized"] = accuracies(nodes_centralized)

nodes_regularized = deepcopy(nodes)
results["regularized"] = regularized_local_FW(nodes_regularized, base_clfs, nb_iter=NB_ITER, beta=BETA, mu=MU, callbacks=callbacks)
hist_accuracies["regularized"] = accuracies(nodes_regularized)

nodes_copy = deepcopy(nodes)
results["async_regularized"] = async_regularized_local_FW(nodes_copy, base_clfs, nb_iter=NB_ITER, beta=BETA, mu=MU, callbacks=callbacks)
hist_accuracies["async_regularized"] = accuracies(nodes_copy)

# nodes_copy = deepcopy(nodes)
# results["local"] = local_FW(nodes_copy, base_clfs, nb_iter=NB_ITER, beta=BETA, callbacks=callbacks)
# hist_accuracies["local"] = accuracies(nodes_copy)

nodes_copy = deepcopy(nodes)
results["global-reg"] = global_regularized_local_FW(nodes_copy, base_clfs, nb_iter=NB_ITER, beta=BETA, callbacks=callbacks)
hist_accuracies["global-reg"] = accuracies(nodes_copy)

# gd_laplacian_nodes = deepcopy(nodes)
# results["gd-regularized-laplacian-1"] = gd_reg_local_FW(gd_laplacian_nodes, base_clfs, gd_method={"name":"laplacian", "pace_gd": 10, "args":(1)}, beta=BETA, nb_iter=NB_ITER, mu=MU, eps=1, callbacks=callbacks)

# gd_laplacian_nodes = deepcopy(nodes)
# results["gd-regularized-laplacian-N"] = gd_reg_local_FW(gd_laplacian_nodes, base_clfs, gd_method={"name":"laplacian", "pace_gd": 10, "args":(N)}, beta=BETA, nb_iter=NB_ITER, mu=MU, eps=1, callbacks=callbacks)

# gd_laplacian_nodes = deepcopy(nodes)
# results["gd-regularized-laplacian-3"] = gd_reg_local_FW(gd_laplacian_nodes, base_clfs, gd_method={"name":"laplacian", "pace_gd": 10, "args":(3)}, beta=BETA, nb_iter=NB_ITER, mu=MU, eps=1, callbacks=callbacks)

# # lafond method
# nodes_copy = deepcopy(nodes)
# results["lafond"] = lafond_FW(nodes_copy, base_clfs, beta=BETA, nb_iter=NB_ITER, callbacks=callbacks)

# # colearning results
# results["colearning"], clf_colearning = colearning(N, X, Y, X_test, Y_test, D, NB_ITER, adj_matrix, similarities)

# get results with true thetas
true_graph = true_theta_graph(nodes_copy, theta_true)
acc = central_accuracy(true_graph)

plt.figure(1, figsize=(18, 10))

plt.subplot(231)
plt.xlabel('nb iterations')
plt.ylabel('train accuracy')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['accuracy'][0] for r in r_list], label='{}'.format(k))
# add results of true thetas
plt.plot(range(len(r_list)), [acc[0]]*len(r_list), label='true-theta')
plt.legend(loc="lower right")

plt.subplot(232)
plt.xlabel('nb iterations')
plt.ylabel('test accuracy')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['accuracy'][1] for r in r_list], label='{}'.format(k))
plt.plot(range(len(r_list)), [acc[1]]*len(r_list), label='true-theta')
plt.legend(loc="lower right")

plt.subplot(233)
plt.xlabel('nb iterations')
plt.ylabel('loss')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [r['loss'] for r in r_list], label='{}'.format(k))
    except:
        pass

plt.legend(loc="lower right")

plt.subplot(234)
plt.xlabel('nb iterations')
plt.ylabel('duality gap')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [r['duality-gap'] for r in r_list], label='{}'.format(k))
    except:
        pass

plt.legend(loc="lower right")

plt.figure(2)
plt.suptitle("Histograms Train Accuracies")

for i, (k, r_list) in enumerate(hist_accuracies.items()):

    plt.subplot(231 + i)
    plt.title(k)
    plt.ylim(0, N)
    plt.hist(r_list[0], 10, range=(0, 1))

plt.figure(3)
plt.suptitle("Histograms Test Accuracies")

for i, (k, r_list) in enumerate(hist_accuracies.items()):

    plt.subplot(231 + i)
    plt.title(k)
    plt.ylim(0, N)
    plt.hist(r_list[1], 10, range=(0, 1))

plt.figure(2, figsize=(18, 10))

max_nb_edges = N*(N-1)

plt.subplot(221)

plt.xlabel('nb iterations')
plt.ylabel('nb edges')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [sum(r['edges']) for r in r_list], label='{}'.format(k))
    except:
        pass

plt.plot(range(len(r_list)), [max_nb_edges]*len(r_list), label='full graph')

plt.legend(loc='center right')

plt.subplot(222)

plt.xlabel('nb iterations')
plt.ylabel('mean nb edges')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [mean(r['edges']) for r in r_list], label='{}'.format(k))
    except:
        pass

plt.plot(range(len(r_list)), [N-1]*len(r_list), label='full graph')

plt.legend(loc='center right')

plt.subplot(223)

plt.xlabel('nb iterations')
plt.ylabel('minimal nb edges')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [min(r['edges']) for r in r_list], label='{}'.format(k))
    except:
        pass

plt.plot(range(len(r_list)), [N-1]*len(r_list), label='full graph')


plt.legend(loc='center right')

plt.subplot(224)

plt.xlabel('nb iterations')
plt.ylabel('maximal nb edges')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [max(r['edges']) for r in r_list], label='{}'.format(k))
    except:
        pass

plt.plot(range(len(r_list)), [N-1]*len(r_list), label='full graph')

plt.legend(loc='center right')

plt.show()