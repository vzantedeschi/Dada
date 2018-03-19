from copy import deepcopy
import numpy as np

from sklearn.utils import shuffle

from classification import get_stumps
from evaluation import central_accuracy, central_loss, best_accuracy
from network import graph
from optimization import centralized_FW, regularized_local_FW, local_FW, async_regularized_local_FW, global_regularized_local_FW, gd_reg_local_FW
from related_works import colearning, lafond_FW
from utils import load_mobiact, get_min_max

import matplotlib.pyplot as plt

# set graph of nodes with local personalized data
NB_ITER = 200
MU = 5
BETA = 10

X, Y, X_test, Y_test, adjacency, similarities, nb_nodes = load_mobiact()

D = X[0].shape[1]
B = 5*D
# set graph
nodes = graph(X, Y, X_test, Y_test, nb_nodes, adjacency, similarities)


# set callbacks for optimization analysis
callbacks = {
    'accuracy': [central_accuracy, []],
    'loss': [central_loss, []]
}

results = {}

vmin, vmax = get_min_max(X)
base_clfs = get_stumps(B, D+1, vmin, vmax)

# nodes_centralized = deepcopy(nodes)
# results["centralized"] = centralized_FW(nodes_centralized, base_clfs, beta=BETA, nb_iter=NB_ITER, callbacks=callbacks)

nodes_regularized = deepcopy(nodes)
results["regularized"] = regularized_local_FW(nodes_regularized, base_clfs, beta=BETA, nb_iter=NB_ITER, mu=MU, callbacks=callbacks)

# local_nodes = deepcopy(nodes)
# results["local"] = local_FW(local_nodes, base_clfs, beta=BETA, nb_iter=NB_ITER, callbacks=callbacks)

lafond_nodes = deepcopy(nodes)
results["lafond"] = lafond_FW(lafond_nodes, base_clfs, beta=BETA, nb_iter=NB_ITER, callbacks=callbacks)

# colearning results
results["colearning"], clf_colearning = colearning(nb_nodes, X, Y, X_test, Y_test, D, NB_ITER, adjacency, similarities)

gd_nodes = deepcopy(nodes)
results["gd-regularized"] = gd_reg_local_FW(gd_nodes, base_clfs, pace_gd=10, beta=BETA, nb_iter=NB_ITER, mu=MU, callbacks=callbacks)

# get best accuracy on train and test samples
best_train_acc, best_test_acc = best_accuracy(nodes)

plt.figure(1, figsize=(18, 10))

plt.subplot(221)
plt.xlabel('nb iterations')
plt.ylabel('train accuracy')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['accuracy'][0] for r in r_list], label='{}'.format(k))
# add results of baseline
plt.plot(range(len(r_list)), [best_train_acc]*len(r_list), label='best-accuracy')

plt.legend()

plt.subplot(222)
plt.xlabel('nb iterations')
plt.ylabel('test accuracy')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['accuracy'][1] for r in r_list], label='{}'.format(k))
plt.plot(range(len(r_list)), [best_test_acc]*len(r_list), label='best-accuracy')
plt.legend()

plt.subplot(223)
plt.xlabel('nb iterations')
plt.ylabel('loss')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [r['loss'] for r in r_list], label='{}'.format(k))
    except:
        pass

plt.legend()

plt.subplot(224)
plt.xlabel('nb iterations')
plt.ylabel('duality gap')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [r['duality-gap'] for r in r_list], label='{}'.format(k))
    except:
        pass

plt.legend()

plt.show()