from copy import deepcopy
import numpy as np

from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

from classification import RandomClassifier
from evaluation import alpha_variance, central_loss, central_accuracy
from network import random_graph, complete_graph
from optimization import average_FW, local_FW, neighbor_FW, centralized_FW
from related_works import lafond_FW
from utils import load_dense_dataset, load_breast_dataset, generate_models, generate_samples

NB_ITER = 100
N = 10

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
    'train-loss': [central_loss, []]
}

methods = {
    "centralized": centralized_FW, 
    "local": local_FW,
    "average": average_FW
}

results = {}
for k, m in methods.items():

    nodes_copy = deepcopy(nodes)
    results[k] = m(nodes_copy, D, NB_ITER, callbacks=callbacks)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=NB_ITER)
bdt.fit(train_x, train_y)
print(accuracy_score(bdt.predict(train_x), train_y))
print(accuracy_score(bdt.predict(test_x), test_y))

# lafond method
nodes_copy = deepcopy(nodes)
results["lafond"] = lafond_FW(nodes, D, NB_ITER, callbacks=callbacks)

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
    plt.plot(range(x_len), [r['train-loss'] for r in r_list], label='{}'.format(k))

plt.legend()

plt.show()