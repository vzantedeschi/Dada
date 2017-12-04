import numpy as np

from sklearn.mixture import GaussianMixture

from utils import compute_adjacencies

class Node():

    def __init__(self, k, sample, labels, test_sample=None, test_labels=None):
        self.id = k
        self.d = sample.shape[1] 
        self.sample = sample
        self.labels = labels
        self.test_sample = test_sample
        self.test_labels = test_labels

    def predict(self, sample):
        return np.sign(np.inner(sample, self.clf))

    def init_matrices(self, n=None):
        if n:
            self.n = n
        else:
            self.n = len(self.neighbors)
        base_clfs = np.eye(self.n, self.d)
        alpha = np.zeros((self.n, 1))
        self.set_margin_matrix(base_clfs)
        self.set_alpha(alpha)

    def get_neighbors_clfs(self):
        nei_clfs = np.vstack([n.clf for n in self.neighbors])
        return nei_clfs

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def set_margin_matrix(self, base_clfs):
        # set margin matrix A
        self.base_clfs = base_clfs
        self.margin = np.inner(self.sample, base_clfs) * self.labels[:, np.newaxis]

    def set_alpha(self, alpha):
        assert alpha.shape == (len(self.base_clfs), 1), alpha.shape
        self.alpha = alpha
        self.clf = np.dot(self.alpha.T, self.base_clfs)
        assert self.clf.shape == (1, self.d)

def centralize_data(nodes):

    if len(nodes) == 1:
        return nodes[0]

    # centralize data
    x, y, x_test, y_test = [], [], [], []
    for n in nodes:
        x.append(n.sample)
        y.append(n.labels)
        x_test.append(n.test_sample)
        y_test.append(n.test_labels)

    node = Node(len(nodes), np.vstack(x), np.concatenate(y), np.vstack(x_test), np.concatenate(y_test))

    return node

# --------------------------------------------------------------- NETWORK CONSTRUCTORS

# line network
def line_network(x, y, nb_nodes=3, cluster_data=False):
    M, _ = x.shape

    # add offset dim
    x_copy = np.c_[x, np.ones(M)]

    if cluster_data:
        gm = GaussianMixture(nb_nodes, init_params="random")
        gm.fit(x_copy)
        labels = gm.predict(x_copy)
        groups = [[x_copy[labels==i], y[labels==i]] for i in range(nb_nodes)]

    else:
        s = M // nb_nodes   
        groups = [[x_copy[i*s:(i+1)*s], y[i*s:(i+1)*s]] for i in range(nb_nodes)]


    nodes = list()
    nei_ids = list()
    for i in range(nb_nodes):

        n = Node(i, *groups[i])
        nei_ids.append([j for j in [i-1, i+1] if j >= 0 and j < nb_nodes])
        nodes.append(n)

    for ids, n in zip(nei_ids, nodes):
        n.set_neighbors([nodes[i] for i in ids])

    return nodes

def synthetic_graph(x, y, x_test, y_test, nb_nodes, theta_true):

    adj_matrix = compute_adjacencies(theta_true, nb_nodes)

    nodes = list()
    nei_ids = list()
    for i in range(nb_nodes):
        # add offset dim
        M, _ = x[i].shape
        x_copy = np.c_[x[i], np.ones(M)]
        M, _ = x_test[i].shape
        x_test_copy = np.c_[x_test[i], np.ones(M)]

        n = Node(i, x_copy, y[i], x_test_copy, y_test[i])
        nei_ids.append([j for j, a in enumerate(adj_matrix[i]) if a != 0])
        nodes.append(n)

    for ids, n in zip(nei_ids, nodes):
        n.set_neighbors([nodes[i] for i in ids])

    return nodes