from math import log

import numpy as np

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

def cost_function(theta, x, y, alpha, max_samples_per_node):
    """Global cost function"""
    return (alpha * np.sum(theta * L.dot(theta))/2
            + (1-alpha)*np.sum([d_i*F(s, x_i, y_i, max_samples_per_node) for d_i, s, x_i, y_i in zip(d, theta, x, y)]))

def cost_function_gradient(L, d, theta, x, y, alpha, max_samples_per_node):
    """Global cost function gradient"""
    return (alpha * L.dot(theta) + (1 - alpha) * np.array([d_i * dF(s, x_i, y_i, max_samples_per_node) for d_i, s, x_i, y_i in zip(d, theta, x, y)]))

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

def colearning(nb_nodes, x, y, x_test, y_test, dim, nb_iter, adjacency, similarities, mu=1, max_samples_per_node=20, checkevery=1):

    results = []
    alpha = 1 / (1 + mu)
    L, d = compute_graph_matrices(nb_nodes, adjacency, similarities)

    theta = np.zeros((nb_nodes, dim))
    results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})
    
    # Collaborative learning
    for t in range(nb_iter):
        theta -= cost_function_gradient(L, d, theta, x, y, alpha, max_samples_per_node)

        if t % checkevery == 0:
            results.append({"train-accuracy": class_ratio(theta, x, y), "test-accuracy": class_ratio(theta, x_test, y_test)})

    return results, theta