import numpy as np

class Node():

    def __init__(self, k, sample, labels):
        self.id = k
        self.sample = sample
        self.labels = labels

    def predict(self, sample):
        return np.sign(np.inner(sample, self.clf))

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_margin_matrix(self, base_clfs):
        # set margin matrix A
        self.base_clfs = base_clfs
        self.alpha = np.zeros((len(base_clfs), 1))
        self.margin = np.inner(self.sample, base_clfs) * self.labels[:, np.newaxis]

    def set_alpha(self, alpha):
        assert alpha.shape == (len(self.base_clfs), 1), alpha.shape
        self.alpha = alpha
        self.clf = np.dot(self.alpha.T, self.base_clfs)

# line network
def line_network(x, y, nb_nodes=3):
    M, _ = x.shape
    s = M//nb_nodes

    nodes = list()
    for i in range(nb_nodes):

        n = Node(i, x[i*s:(i+1)*s], y[i*s:(i+1)*s])
        n.set_neighbors([j for j in [i-1, i+1] if j >= 0 and j < nb_nodes])
        nodes.append(n)

    return nodes



# def get_network_cst_valency(nb_nodes, valency):
#     assert valency < nb_nodes

#     nodes = [Node(k) for k in range(nb_nodes)]