import numpy as np

from sklearn.utils import shuffle

from evaluation import alpha_variance, mean_accuracy
from network import line_network, synthetic_graph
from optimization import average_FW
from utils import load_wine_dataset, generate_models, generate_samples

# X, Y = load_wine_dataset()
# X, Y = shuffle(X, Y)
# M, D = X.shape

# # set network
# nodes = line_network(X, Y, 3, cluster_data=True)

D = 50
random_state = 1
V, theta_true, cluster_indexes = generate_models(random_state=random_state)
_, X, Y, _, _, _, _ = generate_samples(V, theta_true, D, random_state=random_state)

# set graph
nodes = synthetic_graph(X, Y, V, theta_true)

# set base classifiers
N = 50
base_clfs = np.eye(N, D)

# get margin matrices A
for n in nodes:
    n.set_margin_matrix(base_clfs)

# global consensus
average_FW(nodes, 100)

# check convergence
var = alpha_variance(nodes)
print("mean alpha variance", np.mean(var))
# assert np.allclose(var, np.zeros((N,1))), var

# check accuracy
print("train accuracy", mean_accuracy(nodes))