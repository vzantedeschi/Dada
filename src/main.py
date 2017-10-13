import numpy as np

from sklearn.utils import shuffle

from evaluation import alpha_variance, mean_accuracy
from network import line_network
from optimization import average_FW
from utils import load_wine_dataset

X, Y = load_wine_dataset()
X, Y = shuffle(X, Y)
M, D = X.shape

# set network
nodes = line_network(X, Y, 3, cluster_data=True)

# set base classifiers
N = 3
base_clfs = np.eye(N, D)

# get margin matrices A
for n in nodes:
    n.set_margin_matrix(base_clfs)

# global consensus
average_FW(nodes, 100)

# check convergence
var = alpha_variance(nodes)
print(var)
# assert np.allclose(var, np.zeros((N,1))), var

# check accuracy
print(mean_accuracy(nodes))