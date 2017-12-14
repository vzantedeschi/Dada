from copy import deepcopy
import numpy as np

from sklearn.utils import shuffle

from evaluation import alpha_variance, loss, mean_accuracy
from network import complete_graph, synthetic_graph
from optimization import average_FW, local_FW, neighbor_FW, centralized_FW
from utils import load_breast_dataset, generate_models, generate_samples

NB_ITER = 100
N = 2
# D = 10
# random_state = 20160922
# V, theta_true, cluster_indexes = generate_models(nb_clust=1, nodes_per_clust=N, random_state=random_state)
# _, X, Y, X_test, Y_test, _, _ = generate_samples(V, theta_true, D, random_state=random_state)
# set graph
# nodes = synthetic_graph(X, Y, X_test, Y_test, V, theta_true)

X, Y = load_breast_dataset()
M, D = X.shape
nodes = complete_graph(X, Y, nb_nodes=N, cluster_data=False)

# set callbacks for optimization analysis
callbacks = {
    'mean-accuracy': [mean_accuracy, []],
    'train-loss': [loss, []]
}

methods = {
    "centralized": centralized_FW, 
    "local": local_FW,
    "average": average_FW,
    "neighbor": neighbor_FW
}

results = {}
for k, m in methods.items():

    nodes_copy = deepcopy(nodes)
    results[k] = m(nodes_copy, D, NB_ITER, callbacks=callbacks)
    print(len(nodes_copy))

import matplotlib.pyplot as plt

plt.figure(1)
plt.xlabel('nb iterations')
plt.ylabel('train accuracy')

for k, r_list in results.items():
    plt.plot(range(NB_ITER), [r['mean-accuracy'][0] for r in r_list], label='{}'.format(k))

plt.legend()

plt.figure(2)
plt.xlabel('nb iterations')
plt.ylabel('test accuracy')

for k, r_list in results.items():
    plt.plot(range(NB_ITER), [r['mean-accuracy'][1] for r in r_list], label='{}'.format(k))

plt.legend()

plt.figure(3)
plt.xlabel('nb iterations')
plt.ylabel('train loss')

for k, r_list in results.items():
    plt.plot(range(NB_ITER), [r['train-loss'] for r in r_list], label='{}'.format(k))

plt.legend()

plt.show()