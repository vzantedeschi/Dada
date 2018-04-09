from copy import deepcopy
import numpy as np
from statistics import mean

from sklearn.utils import shuffle

from classification import get_stumps
from evaluation import central_accuracy, central_loss, best_accuracy, edges
from network import line_network, synthetic_graph, true_theta_graph
from optimization import centralized_FW, regularized_local_FW, local_FW, async_regularized_local_FW, global_regularized_local_FW, gd_reg_local_FW, async_gd_reg_local_FW
from related_works import colearning, lafond_FW
from utils import generate_models, generate_moons, get_min_max

import matplotlib.pyplot as plt

# set graph of nodes with local personalized data
NB_ITER = 300
N = 20
D = 20
B = 200
NOISE_R = 0.05
random_state = 2017
BETA = 10
MU = 1

max_nb_edges = N*(N-1)
V, theta_true, cluster_indexes = generate_models(nb_clust=1, nodes_per_clust=N, random_state=random_state)
_, X, Y, X_test, Y_test, max_nb_instances = generate_moons(V, theta_true, D, random_state=random_state, sample_error_rate=NOISE_R)

# set graph
nodes, adj_matrix, similarities = synthetic_graph(X, Y, X_test, Y_test, V, theta_true, max_nb_instances)

# set callbacks for optimization analysis
callbacks = {
    'accuracy': [central_accuracy, []],
    'loss': [central_loss, []],
    'edges': [edges, []],
}

results = {}

vmin, vmax = get_min_max(X)
base_clfs = get_stumps(n=B, d=D+1, min_v=vmin, max_v=vmax)

# gd_knn_nodes = deepcopy(nodes)
# results["gd-regularized-knn"] = gd_reg_local_FW(gd_knn_nodes, base_clfs, gd_method={"name":"knn", "pace_gd": 20, "args":(N//2)}, beta=BETA, nb_iter=NB_ITER, mu=MU, reset_step=False, callbacks=callbacks)

# nodes_copy = deepcopy(nodes)
# results["async-gd"] = async_gd_reg_local_FW(nodes_copy, base_clfs, beta=BETA, nb_iter=NB_ITER,pace_gd=20, callbacks=callbacks)

# gd_laplacian_nodes = deepcopy(nodes)
# results["gd-regularized-laplacian-1"] = gd_reg_local_FW(gd_laplacian_nodes, base_clfs, gd_method={"name":"laplacian", "pace_gd": 20, "args":(1)}, beta=BETA, nb_iter=NB_ITER, mu=MU, eps=N/2, callbacks=callbacks)

nodes_regularized = deepcopy(nodes)
results["regularized"] = regularized_local_FW(nodes_regularized, base_clfs, beta=BETA, nb_iter=NB_ITER, mu=MU, callbacks=callbacks)

nodes_regularized = deepcopy(nodes)
results["async-regularized"] = async_regularized_local_FW(nodes_regularized, base_clfs, beta=1, nb_iter=NB_ITER, mu=MU, callbacks=callbacks)

# nodes_local = deepcopy(nodes)
# results["local"] = local_FW(nodes_local, base_clfs, beta=BETA, nb_iter=NB_ITER, callbacks=callbacks)

# gd_fullknn_nodes = deepcopy(nodes)
# results["gd-regularized-fullknn"] = gd_reg_local_FW(gd_fullknn_nodes, base_clfs, gd_method={"name":"full-knn", "pace_gd": 20, "args":(N//2)}, beta=BETA, nb_iter=NB_ITER, mu=MU, reset_step=False, callbacks=callbacks)

# lafond_nodes = deepcopy(nodes)
# results["lafond"] = lafond_FW(lafond_nodes, base_clfs, nb_iter=NB_ITER, beta=BETA, callbacks=callbacks)

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

plt.legend(loc='lower right')

plt.subplot(222)
plt.xlabel('nb iterations')
plt.ylabel('test accuracy')

for k, r_list in results.items():
    plt.plot(range(len(r_list)), [r['accuracy'][1] for r in r_list], label='{}'.format(k))
plt.plot(range(len(r_list)), [best_test_acc]*len(r_list), label='best-accuracy')

plt.legend(loc='lower right')

plt.subplot(223)
plt.xlabel('nb iterations')
plt.ylabel('loss')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [r['loss'] for r in r_list], label='{}'.format(k))
    except:
        pass

plt.subplot(224)
plt.xlabel('nb iterations')
plt.ylabel('duality gap')

for k, r_list in results.items():
    try:
        plt.plot(range(len(r_list)), [r['duality-gap'] for r in r_list], label='{}'.format(k))
    except:
        pass

plt.legend()

plt.figure(2, figsize=(18, 10))

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

# for NODE in range(N):

#     print(NODE)
#     plt.figure(NODE+2, figsize=(16, 5))
#     plt.suptitle(NODE)
#     # our method
#     plt.subplot(221)
#     plt.title("local FW")
#     # training data
#     X = local_nodes[NODE].sample
#     Y = local_nodes[NODE].labels

#     X_test = local_nodes[NODE].test_sample
#     Y_test = local_nodes[NODE].test_labels

#     # construct grid
#     x_min,x_max = X_test[:,0].min() - 0.2, X_test[:,0].max() + 0.2
#     y_min, y_max = X_test[:,1].min() - 0.2, X_test[:,1].max() + 0.2
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

#     # expand dimensions
#     grid_set = np.c_[xx.ravel(), yy.ravel()]
#     grid_set = np.hstack((grid_set, np.zeros((len(grid_set), D - 1))))
#     y = local_nodes[NODE].predict(grid_set).reshape(xx.shape)

#     plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, linewidths=10)
#     plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=plt.cm.coolwarm)

#     plt.contourf(xx, yy, y, cmap=plt.cm.coolwarm, alpha=0.2)

#     # colearning
#     plt.subplot(222)
#     plt.title("colearning")

#     # training data
#     X = local_nodes[NODE].sample
#     Y = local_nodes[NODE].labels

#     grid_set = np.c_[xx.ravel(), yy.ravel()]
#     grid_set = np.hstack((grid_set, np.zeros((len(grid_set), D - 2))))
#     y = np.sign(np.inner(grid_set, clf_colearning[NODE, :])).reshape(xx.shape)

#     plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, linewidths=10)
#     plt.contourf(xx, yy, y, cmap=plt.cm.coolwarm, alpha=0.2)
#     plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=plt.cm.coolwarm)

#     plt.subplot(223)
#     plt.title("regularized FW")

#     # training data
#     X = nodes_regularized[NODE].sample
#     Y = nodes_regularized[NODE].labels

#     X_test = nodes_regularized[NODE].test_sample
#     Y_test = nodes_regularized[NODE].test_labels

#     # expand dimensions
#     grid_set = np.c_[xx.ravel(), yy.ravel()]
#     grid_set = np.hstack((grid_set, np.zeros((len(grid_set), D - 1))))
#     y = nodes_regularized[NODE].predict(grid_set).reshape(xx.shape)

#     plt.scatter(X[:,0], X[:,1], c=Y, cmap=plt.cm.coolwarm, linewidths=10)
#     plt.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=plt.cm.coolwarm)

#     plt.contourf(xx, yy, y, cmap=plt.cm.coolwarm, alpha=0.2)

#     break

plt.show()