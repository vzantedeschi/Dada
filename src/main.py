import numpy as np

from sklearn.utils import shuffle

from evaluation import alpha_variance, mean_accuracy
from network import line_network, synthetic_graph
from optimization import average_FW, local_FW
from utils import load_wine_dataset, generate_models, generate_samples

NB_ITER = 100
# X, Y = load_wine_dataset()
# X, Y = shuffle(X, Y)
# M, D = X.shape

# # set network
# nodes = line_network(X, Y, 3, cluster_data=True)

D = 50
random_state = 1
V, theta_true, cluster_indexes = generate_models(random_state=random_state)
_, X, Y, X_test, Y_test, _, _ = generate_samples(V, theta_true, D, random_state=random_state)

# set graph
nodes = synthetic_graph(X, Y, X_test, Y_test, V, theta_true)

# set base classifiers
N = 50

# get margin matrices A
for n in nodes:
    n.init_matrices(N)

# set callbacks for optimization analysis
callbacks = {
    'alpha-variance': alpha_variance,
    'mean-accuracy': mean_accuracy
}


# local learning
results_local = local_FW(nodes, NB_ITER, callbacks=callbacks)

# global consensus
results_global = average_FW(nodes, NB_ITER, callbacks=callbacks)   

import matplotlib.pyplot as plt

plt.figure(1)
plt.xlabel('nb iterations')
plt.ylabel('mean accuracy')
plt.plot(range(NB_ITER), [r['mean-accuracy'][0] for r in results_global], label='avg global train accuracy')
plt.plot(range(NB_ITER), [r['mean-accuracy'][1] for r in results_global], label='avg global test accuracy')
plt.plot(range(NB_ITER), [r['mean-accuracy'][0] for r in results_local], label='avg local train accuracy')
plt.plot(range(NB_ITER), [r['mean-accuracy'][1] for r in results_local], label='avg local test accuracy')
plt.legend()

plt.figure(2)
plt.xlabel('nb iterations')
plt.ylabel('alpha variance')
plt.plot(range(NB_ITER), [r['alpha-variance'] for r in results_global], label='global alpha variance')
plt.plot(range(NB_ITER), [r['alpha-variance'] for r in results_local], label='local alpha variance')

plt.legend()

plt.show()