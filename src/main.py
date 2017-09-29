import numpy as np

from sklearn.datasets import load_wine
from sklearn.utils import shuffle

from evaluation import alpha_variance, mean_accuracy
from network import line_network
from optimization import average_FW

# load dataset
X, Y = load_wine(return_X_y=True)
X, Y = shuffle(X, Y)
M, D = X.shape

# set network
nodes = line_network(X, Y, 4)

# set base classifiers
N = 3
base_clfs = np.eye(N, D)

# get margin matrices A
for n in nodes:
    n.set_margin_matrix(base_clfs)

# global consensus
average_FW(nodes, 10)

# check convergence
var = alpha_variance(nodes)
assert np.allclose(var, np.zeros((N,1)))

# check accuracy
print(mean_accuracy(nodes))