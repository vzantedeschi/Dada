from math import log

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

from utils import kalo_utils

# ----------------------------------------------------------------------- LAFOND

def minimize_gradients(gradients, nodes, gamma, beta=None):

    new_alphas = []
    duals = []

    for n, g in zip(nodes, gradients):

        if beta is None:
            # simplex constraint
            j = np.argmin(g)
            s_k = np.asarray([[1] if h==j else [0] for h in range(n.n)])
        else:
            # l1 constraint
            j = np.argmax(abs(g))
            s_k = np.sign(-g[j, :]) * beta * np.asarray([[1] if h==j else [0] for h in range(n.n)])

        alpha_k = (1 - gamma) * n.alpha + gamma * s_k
        new_alphas.append(alpha_k)
        duals.append(np.dot((alpha_k - s_k).squeeze(), g.squeeze()))

    return new_alphas, duals

def gac_routine(vectors, nodes, nb_iter):

    # update gradients using GAC routine
    new_vectors = []
    
    for _ in range(nb_iter):

        new_vectors = []

        for n in nodes:

            new_vectors.append(np.sum([s*vectors[m.id] for m, s in zip(n.neighbors, n.sim)] + [vectors[n.id]], axis=0) / (1+n.sum_similarities))

    return new_vectors

def lafond_FW(nodes, base_clfs, nb_iter=1, beta=None, c1=5, t=1, monitors=None):

    results = []
    
    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    results.append({})  
    for k, call in monitors.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = 0

    # frank-wolfe
    for i in range(nb_iter):

        nb_iter_gac = int(c1 + log(i+1))

        gamma = 2 / (2 + i)

        # compute init gradients
        gradients = []

        for n in nodes:

            w = n.compute_weights(t)
            g = - np.dot(n.margin.T, w)
            
            gradients.append(g)

        gradients = gac_routine(gradients, nodes, nb_iter_gac)

        alphas, duals = minimize_gradients(gradients, nodes, gamma, beta)

        alphas = gac_routine(alphas, nodes, nb_iter_gac)

        for n, a in zip(nodes, alphas):
            n.set_alpha(a)

        results.append({})  
        for k, call in monitors.items():
            results[i+1][k] = call[0](nodes, *call[1])
        results[i+1]["duality-gap"] = sum(duals)

    return results

# -------------------------------------------------------------------- COLEARNING
# code adapted from gitlab.inria.fr/magnet/decentralizedpaper/blob/master/notebooks/colearning.ipynb

# Functions definitions
def F(s, x_i, y_i, max_samples_per_node):
    """Local loss function"""
    return (1 - x_i.dot(s) * y_i).clip(min=0).sum() / max_samples_per_node

def dF(s, x_i, y_i, max_samples_per_node):
    """Local loss function gradient"""
    return - x_i.T.dot(y_i * ((1 - x_i.dot(s) * y_i) > 0)) / max_samples_per_node

def cost_function(L, d, theta, x, y, mu, max_samples_per_node):
    """Global cost function"""
    return (mu * np.sum(theta * L.dot(theta))/2
            + np.sum([d_i*F(s, x_i, y_i, max_samples_per_node) for d_i, s, x_i, y_i in zip(d, theta, x, y)]))

def cost_function_gradient(L, d, theta, x, y, mu, max_samples_per_node):
    """Global cost function gradient"""
    return (mu * L.dot(theta) + np.array([d_i * dF(s, x_i, y_i, max_samples_per_node) for d_i, s, x_i, y_i in zip(d, theta, x, y)]))

# Local initial models
def compute_theta_loc(n, x, y, dim, max_samples_per_node):
    """Compute local models
    (naive gradient descent)"""

    theta_loc = np.zeros((n, dim))
    for _ in range(10):
        theta_loc -= np.array([dF(s, x_i, y_i, max_samples_per_node) for s, x_i, y_i in zip(theta_loc, x, y)])

    return theta_loc

def class_ratio(theta, x, y):
    """Classification succes rate on given samples"""
    return np.concatenate([x_i.dot(s) * y_i > 0 for s, x_i, y_i in zip(theta, x, y)]).mean()

def class_ratio_pernode(theta, x, y):
    """Classification succes rate on given samples for each agent"""
    return np.array([np.mean(x_i.dot(s) * y_i > 0) for s, x_i, y_i in zip(theta, x, y)])

def compute_graph_matrices(n, adjacency, similarities):
    """Compute graph matrices according to true models of agents"""
    
    I = np.eye(n)
    W = similarities * adjacency
    W /= W.sum(axis=1).mean() # Normalize W
    d = W.sum(axis=1)
    D = np.diag(d)
    L = D - W

    return L, d

def obj_kalo(w, z, S, l, mu, la):

    d = S.dot(w)

    if np.count_nonzero(d) < len(d):
        return np.inf

    return d.dot(l) + (mu/2) * (w.dot(z) - np.log(d).sum() + la * (mu/2) * w.dot(w))

def graph_discovery(nb_nodes, theta, similarities, S, triu_ix, l, mu=1, la=1, *args):

    n_pairs = nb_nodes * (nb_nodes - 1) // 2
    stop_thresh = 10e-2 / n_pairs

    z = pairwise_distances(theta)**2
    z = z[triu_ix]

    if similarities is not None:
        w = np.asarray(similarities[triu_ix])
    else:
        w = np.ones(n_pairs)
    d = S.dot(w)

    gamma = 1 / (np.linalg.norm(l.dot(S)) + (mu/2) * (np.linalg.norm(z) + np.linalg.norm(S.T.dot(S)) + 2 * la * mu / 2))
    obj = obj_kalo(w, z, S, l, mu, la)
# 
    # print('\n', 0, obj)
    for k in range(2000):

        grad = l.dot(S) + (mu / 2) * (z - (1. / d).dot(S) + 2 * la * (mu / 2) * w)

        new_w = w - gamma * grad
        new_w[new_w < 0] = 0

        new_obj = obj_kalo(new_w, z, S, l, mu, la)
        # print(gamma, new_obj)
        if new_obj > obj:
            gamma /= 2
            # print("inf")

        elif abs(obj - new_obj) > abs(stop_thresh * obj):
            obj = new_obj
            w = new_w
            gamma *= 1.05
            # print(k, obj)

        else:
            w = new_w
            break
        
        d = S.dot(w)

    # print(k, new_obj)

    # print("done in", k)
    similarities = np.zeros((nb_nodes, nb_nodes))
    similarities[triu_ix] = similarities.T[triu_ix] = w

    return similarities

def block_graph_discovery(nb_nodes, theta, similarities, S, triu_ix, l, map_idx, mu=1, la=1, kappa=1, max_iter=1e6, *args):

    n_pairs = nb_nodes * (nb_nodes - 1) // 2
    stop_thresh = 10e-2 / n_pairs

    z = pairwise_distances(theta)**2
    z = z[triu_ix]

    if similarities is not None:
        w = np.asarray(similarities[triu_ix])
    else:
        w = 0.01 * (1 / np.maximum(z, 1))

    gamma = 0.5

    obj = obj_kalo(w, z, S, l, mu, la)

    new_w = w.copy()

    # print('\n', "it=", 0, "obj=", obj, "gamma=", gamma)
    for k in range(int(max_iter)):

        rnd_j = np.random.choice(nb_nodes, 1+kappa, replace=False)
        i, others = rnd_j[0], rnd_j[1:]

        idx_block = map_idx[np.minimum(i, others), np.maximum(i, others)]
        d_block = S[rnd_j, :].dot(new_w)
        S_block = S[rnd_j, :][:, idx_block]

        grad = l[rnd_j].dot(S_block) + (mu / 2) * (z[idx_block] - (1. / d_block).dot(S_block) + 2 * la * (mu / 2) * new_w[idx_block])

        new_w[idx_block] = new_w[idx_block] - gamma * grad
        new_w[new_w < 0] = 0

        new_obj = obj_kalo(new_w, z, S, l, mu, la)

        if k % nb_nodes == 0:

            if new_obj > obj or not np.isfinite(new_obj):
                gamma /= 2
                new_w = w.copy()
                new_obj = obj

            elif obj - new_obj < abs(obj) / stop_thresh:
                break

            else:
                gamma *= 1.05
                w = new_w.copy()
                obj = new_obj

    print(k)

    # print("done in", k)
    similarities = np.zeros((nb_nodes, nb_nodes))
    similarities[triu_ix] = similarities.T[triu_ix] = w

    return similarities

def local_colearning(nb_nodes, x, y, x_test, y_test, dim, nb_iter, mu=1, max_samples_per_node=1, checkevery=1):

    results = []

    L, d = compute_graph_matrices(nb_nodes, np.eye(nb_nodes), np.eye(nb_nodes))

    theta = np.zeros((nb_nodes, dim))
    results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})
    
    # Collaborative learning
    for t in range(nb_iter):
        theta -= cost_function_gradient(L, d, theta, x, y, mu, max_samples_per_node)

        if t % checkevery == 0:
            results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})

    return results, theta

def colearning(nb_nodes, x, y, x_test, y_test, dim, nb_iter, adjacency, similarities, mu=1, max_samples_per_node=1, checkevery=1):

    results = []

    L, d = compute_graph_matrices(nb_nodes, adjacency, similarities)

    theta = np.zeros((nb_nodes, dim))
    results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})
    
    # Collaborative learning
    for t in range(nb_iter):
        theta -= cost_function_gradient(L, d, theta, x, y, mu, max_samples_per_node)

        if t % checkevery == 0:
            results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})

    return results, theta

def block_alternating_colearning(nb_nodes, x, y, x_test, y_test, dim, nb_iter, mu=1, la=1, kappa=1, max_samples_per_node=1, pace_gd=10, max_iter_gd=1e6, checkevery=1):

    results = []

    S, triu_ix, map_idx = kalo_utils(nb_nodes)
    
    # init with graph learned from local models
    _, theta = local_colearning(nb_nodes, x, y, x_test, y_test, dim, nb_iter, mu, max_samples_per_node, checkevery=nb_iter)
    l = np.asarray([F(s, x_i, y_i, max_samples_per_node) for s, x_i, y_i in zip(theta, x, y)])

    similarities = block_graph_discovery(nb_nodes, theta, None, S, triu_ix, l, map_idx, mu=mu, la=la, kappa=kappa, max_iter=max_iter_gd)
    adjacency = similarities > 0
    L, d = compute_graph_matrices(nb_nodes, adjacency, similarities)

    theta = np.zeros((nb_nodes, dim))
    results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})
    
    # Collaborative learning
    for t in range(1, nb_iter+1):

        # print(cost_function(L, d, theta, x, y, mu, max_samples_per_node))

        theta -= cost_function_gradient(L, d, theta, x, y, mu, max_samples_per_node)

        if t % pace_gd == 0:

            l = np.asarray([F(s, x_i, y_i, max_samples_per_node) for s, x_i, y_i in zip(theta, x, y)])
            similarities = block_graph_discovery(nb_nodes, theta, similarities, S, triu_ix, l, map_idx, mu=mu, la=la, kappa=kappa, max_iter=max_iter_gd)
            adjacency = similarities > 0
            L, d = compute_graph_matrices(nb_nodes, adjacency, similarities)

        if t % checkevery == 0:
            results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})

    return results, theta

def alternating_colearning(nb_nodes, x, y, x_test, y_test, dim, nb_iter, mu=1, la=1, max_samples_per_node=1, pace_gd=10, checkevery=1):

    results = []

    S, triu_ix, _ = kalo_utils(nb_nodes)
    
    # init with graph learned from local models
    _, theta = local_colearning(nb_nodes, x, y, x_test, y_test, dim, nb_iter, mu, max_samples_per_node, checkevery=nb_iter)
    l = np.asarray([F(s, x_i, y_i, max_samples_per_node) for s, x_i, y_i in zip(theta, x, y)])

    similarities = graph_discovery(nb_nodes, theta, None, S, triu_ix, l, mu=mu, la=la)
    adjacency = similarities > 0
    L, d = compute_graph_matrices(nb_nodes, adjacency, similarities)

    theta = np.zeros((nb_nodes, dim))
    results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})
    
    # Collaborative learning
    for t in range(1, nb_iter+1):

        # print(cost_function(L, d, theta, x, y, mu, max_samples_per_node))

        theta -= cost_function_gradient(L, d, theta, x, y, mu, max_samples_per_node)

        if t % pace_gd == 0:

            l = np.asarray([F(s, x_i, y_i, max_samples_per_node) for s, x_i, y_i in zip(theta, x, y)])
            similarities = graph_discovery(nb_nodes, theta, similarities, S, triu_ix, l, mu=mu, la=la)
            adjacency = similarities > 0
            L, d = compute_graph_matrices(nb_nodes, adjacency, similarities)

        if t % checkevery == 0:
            results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})

    return results, theta