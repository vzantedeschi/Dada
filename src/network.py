import numpy as np

from sklearn.mixture import GaussianMixture

from utils import compute_adjacencies

class Node():

    def __init__(self, k, sample, labels, test_sample=None, test_labels=None):
        self.id = k
        self.sample = sample
        self.labels = labels
        self.test_sample = test_sample
        self.test_labels = test_labels

    def predict(self, sample):
        return np.sign(np.inner(sample, self.clf))

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_margin_matrix(self, base_clfs):
        # set margin matrix A
        self.base_clfs = base_clfs
        self.alpha = np.zeros((len(base_clfs), 1))
        self.margin = np.inner(self.sample, base_clfs) * self.labels[:, np.newaxis]
        assert self.margin.shape == (len(self.sample), (len(base_clfs))), (self.alpha.shape, base_clfs.shape, self.margin.shape)

    def set_alpha(self, alpha):
        assert alpha.shape == (len(self.base_clfs), 1), alpha.shape
        self.alpha = alpha
        self.clf = np.dot(self.alpha.T, self.base_clfs)

# line network
def line_network(x, y, nb_nodes=3, cluster_data=False):
    M, _ = x.shape

    if cluster_data:
        gm = GaussianMixture(nb_nodes, init_params="random")
        gm.fit(x)
        labels = gm.predict(x)
        groups = [[x[labels==i], y[labels==i]] for i in range(nb_nodes)]

    else:
        s = M // nb_nodes   
        groups = [[x[i*s:(i+1)*s], y[i*s:(i+1)*s]] for i in range(nb_nodes)]


    nodes = list()
    for i in range(nb_nodes):

        n = Node(i, *groups[i])
        n.set_neighbors([j for j in [i-1, i+1] if j >= 0 and j < nb_nodes])
        nodes.append(n)

    return nodes

def synthetic_graph(x, y, x_test, y_test, nb_nodes, theta_true):

    adj_matrix = compute_adjacencies(theta_true, nb_nodes)

    nodes = list()
    for i in range(nb_nodes):

        n = Node(i, x[i], y[i], x_test[i], y_test[i])
        n.set_neighbors([j for j, a in enumerate(adj_matrix[i]) if a != 0])
        nodes.append(n)

    return nodes


# def get_network_cst_valency(nb_nodes, valency):
#     assert valency < nb_nodes

#     nodes = [Node(k) for k in range(nb_nodes)]