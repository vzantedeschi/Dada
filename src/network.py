import numpy as np

from utils import compute_adjacencies, partition

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
            self.n = 2 * n
        else:
            self.n = len(self.neighbors)
        base_clfs = np.append(np.eye(self.n // 2, self.d), -np.eye(self.n // 2, self.d), axis=0) 
        alpha = np.zeros((self.n, 1))
        self.set_margin_matrix(base_clfs)
        self.set_alpha(alpha)

    def compute_weights(self, temp=1, distr=True):
        w = np.exp(-np.dot(self.margin, self.alpha) / temp)
        if distr:
            w = np.nan_to_num(w / np.sum(w))
        return w

    def get_neighbors_clfs(self):
        nei_clfs = np.vstack([n.clf for n in self.neighbors])
        return nei_clfs

    def set_neighbors(self, neighbors, sim=None):
        self.neighbors = neighbors
        self.sim = sim

    def set_margin_matrix(self, base_clfs):
        # set margin matrix A
        self.base_clfs = base_clfs
        self.margin = np.inner(self.sample, base_clfs) * self.labels[:, np.newaxis]

    def set_alpha(self, alpha):
        assert alpha.shape == (len(self.base_clfs), 1), alpha.shape
        self.alpha = alpha
        self.clf = np.dot(self.alpha.T, self.base_clfs)
        assert self.clf.shape == (1, self.d)

    def set_test_set(self, x, y):
        self.test_sample = x
        self.test_labels = y

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

    try:
        node = Node(len(nodes), np.vstack(x), np.concatenate(y), np.vstack(x_test), np.concatenate(y_test))
    except:
        node = Node(len(nodes), np.vstack(x), np.concatenate(y))

    return node

# --------------------------------------------------------------- NETWORK CONSTRUCTORS

# line network
def line_network(x, y, nb_nodes=3, cluster_data=False):
    M, _ = x.shape
    # add offset dim
    x_copy = np.c_[x, np.ones(M)]

    # clustering
    groups = partition(x_copy, y, nb_nodes, cluster_data)

    nodes = list()
    nei_ids = list()
    for i in range(nb_nodes):

        n = Node(i, *groups[i])
        nei_ids.append([j for j in [i-1, i+1] if j >= 0 and j < nb_nodes])
        nodes.append(n)

    for ids, n in zip(nei_ids, nodes):
        n.set_neighbors([nodes[i] for i in ids])

    return nodes

def complete_graph(x, y, nb_nodes=3, cluster_data=False, rnd_state=None):
    M, _ = x.shape
    # add offset dim
    x_copy = np.c_[x, np.ones(M)]

    # clustering
    groups = partition(x_copy, y, nb_nodes, cluster_data, random_state=None)

    nodes = list()
    for i in range(nb_nodes):

        n = Node(i, *groups[i])
        nodes.append(n)

    for i, n in enumerate(nodes):
        n.set_neighbors([nodes[j] for j in range(nb_nodes) if i!=j])

    return nodes

def synthetic_graph(x, y, x_test, y_test, nb_nodes, theta_true):
    """ edge weight = sim*nb_instances """

    adj_matrix, similarities = compute_adjacencies(theta_true, nb_nodes)

    nodes = list()
    nei_ids = list()
    nei_sim = list()
    for i in range(nb_nodes):

        n = Node(i, x[i], y[i], x_test[i], y_test[i])

        nei_ids.append([])
        nei_sim.append([])
        for j, a in enumerate(adj_matrix[i]):
            if a != 0:
                nei_ids[i].append(j)
                nei_sim[i].append(similarities[i][j])
        nodes.append(n)

    for ids, sims, n in zip(nei_ids, nei_sim, nodes):
        n.set_neighbors([nodes[i] for i in ids], [len(nodes[i].sample)*s for s,i in zip(sims, ids)])

    return nodes

def true_theta_graph(nodes, theta_true):

    new_graph = list()

    for i, n in enumerate(nodes):

        print(theta_true[i], n.clf)
        m = Node(i, n.sample, n.labels, n.test_sample, n.test_labels)
        m.clf = np.append(theta_true[i], np.zeros((1, n.d - 2)))
        new_graph.append(m)

    return new_graph
