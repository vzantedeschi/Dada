from copy import deepcopy
import numpy as np

from sklearn.utils import shuffle

from evaluation import central_accuracy, central_loss, accuracies
from network import line_network, synthetic_graph, true_theta_graph
from optimization import centralized_FW, regularized_local_FW, local_FW, async_regularized_local_FW, global_regularized_local_FW
from related_works import lafond_FW
from utils import generate_models, generate_samples

import matplotlib.pyplot as plt

# set graph of nodes with local personalized data
NB_ITER = 50
N = 100
D = 20
NOISE_R = 0.05
random_state = 2017
V, theta_true, cluster_indexes = generate_models(nb_clust=1, nodes_per_clust=N, random_state=random_state)
_, X, Y, X_test, Y_test, _, _ = generate_samples(V, theta_true, D, random_state=random_state, sample_error_rate=NOISE_R)

# set graph
nodes, adj_matrix, similarities = synthetic_graph(X, Y, X_test, Y_test, V, theta_true)

# set callbacks for optimization analysis
callbacks = {
    'accuracy': [central_accuracy, []],
    'loss': [central_loss, []]
}

results = {}
hist_accuracies = {}

nodes_centralized = deepcopy(nodes)
results["centralized"] = centralized_FW(nodes_centralized, 2*D, nb_iter=NB_ITER, callbacks=callbacks)
hist_accuracies["centralized"] = accuracies(nodes_centralized)

nodes_regularized = deepcopy(nodes)
results["regularized"] = regularized_local_FW(nodes_regularized, 2*D, nb_iter=NB_ITER, mu=1, callbacks=callbacks)
hist_accuracies["regularized"] = accuracies(nodes_regularized)

nodes_copy = deepcopy(nodes)
results["async_regularized"] = async_regularized_local_FW(nodes_copy, 2*D, nb_iter=NB_ITER, mu=1, callbacks=callbacks)
hist_accuracies["async_regularized"] = accuracies(nodes_copy)

nodes_copy = deepcopy(nodes)
results["local"] = local_FW(nodes_copy, 2*D, nb_iter=NB_ITER, callbacks=callbacks)
hist_accuracies["local"] = accuracies(nodes_copy)

nodes_copy = deepcopy(nodes)
results["global-reg"] = global_regularized_local_FW(nodes_copy, 2*D, nb_iter=NB_ITER, callbacks=callbacks)
hist_accuracies["global-reg"] = accuracies(nodes_copy)

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
plt.legend()

plt.subplot(232)
plt.xlabel('nb iterations')
plt.ylabel('test accuracy')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['accuracy'][1] for r in r_list], label='{}'.format(k))
plt.plot(range(len(r_list)), [acc[1]]*len(r_list), label='true-theta')
plt.legend()

plt.subplot(233)
plt.xlabel('nb iterations')
plt.ylabel('loss')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['loss'] for r in r_list], label='{}'.format(k))

plt.legend()

plt.subplot(234)
plt.xlabel('nb iterations')
plt.ylabel('duality gap')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['duality-gap'] for r in r_list], label='{}'.format(k))

plt.legend()

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

plt.show()