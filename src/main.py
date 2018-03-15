from copy import deepcopy
import numpy as np

from sklearn.metrics import accuracy_score

from classification import RandomClassifier, get_basis
from evaluation import central_loss, central_accuracy
from network import random_graph, complete_graph
from optimization import average_FW, local_FW, centralized_FW, regularized_local_FW
from related_works import lafond_FW
from utils import load_dense_dataset, load_breast_dataset, generate_models, generate_samples

NB_ITER = 100
N = 30
BETA = 1
MU = 1

TRAIN_FILE = "datasets/ijcnn1.train"
TEST_FILE = "datasets/ijcnn1.test"

train_x, train_y = load_dense_dataset(TRAIN_FILE)
test_x, test_y = load_dense_dataset(TEST_FILE)
D = test_x.shape[1]

# train_x, train_y = load_breast_dataset()

nodes = random_graph(train_x, train_y, nb_nodes=N, prob_edge=0.1, cluster_data=True)
# nodes = complete_graph(train_x, train_y, nb_nodes=N, cluster_data=False)

# set test set
test_m = test_x.shape[0]
test_x_copy = np.c_[test_x, np.ones(test_m)]
for n in nodes:
    n.set_test_set(test_x_copy, test_y)

# set callbacks for optimization analysis
callbacks = {
    'accuracy': [central_accuracy, []],
    'loss': [central_loss, []]
}

methods = {
    "centralized": centralized_FW, 
    "lafond": lafond_FW,
    "local": local_FW,
    "average": average_FW
}

base_clfs = get_basis(n=D, d=D+1)

results = {}
for k, m in methods.items():

    nodes_copy = deepcopy(nodes)
    results[k] = m(nodes_copy, base_clfs, nb_iter=NB_ITER, beta=BETA, callbacks=callbacks)

# regularized method
nodes_copy = deepcopy(nodes)
results["regularized"] = regularized_local_FW(nodes_copy, base_clfs, nb_iter=NB_ITER, beta=BETA, mu=MU, callbacks=callbacks)

random_clf = RandomClassifier()
train_acc_rnd = random_clf.score(train_x, train_y)
test_acc_rnd = random_clf.score(test_x, test_y)

import matplotlib.pyplot as plt

plt.figure(1)
plt.xlabel('nb iterations')
plt.ylabel('train accuracy')

x_len = len(results["centralized"])

for k, r_list in results.items():
    plt.plot(range(x_len), [r['accuracy'][0] for r in r_list], label='{}'.format(k))
plt.plot(range(x_len), [train_acc_rnd]*x_len, label='random')
plt.legend()

plt.figure(2)
plt.xlabel('nb iterations')
plt.ylabel('test accuracy')

for k, r_list in results.items():
    plt.plot(range(x_len), [r['accuracy'][1] for r in r_list], label='{}'.format(k))
plt.plot(range(x_len), [test_acc_rnd]*x_len, label='random')
plt.legend()

plt.figure(3)
plt.xlabel('nb iterations')
plt.ylabel('train loss')

for k, r_list in results.items():
    plt.plot(range(x_len), [r['loss'] for r in r_list], label='{}'.format(k))

plt.legend()

plt.show()