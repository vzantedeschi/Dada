import numpy as np
from random import choice

from network import centralize_data

"""
Boosting algorithms based on Frank Wolfe optimization
"""

# ----------------------------------------------------------- specific utils


def one_frank_wolfe_round(nodes, gamma, beta=1, t=1, mu=0, reg_sum=None, simplex=True):
    """ Modify nodes!
    """
 
    duals = [float("inf")] * len(nodes)

    for i, n in enumerate(nodes):

        if reg_sum:
            r = reg_sum[i]
        else:
            r = None

        frank_wolfe_on_one_node(n, i, gamma, duals, beta, t, mu, r, simplex)

    return duals

def frank_wolfe_on_one_node(n, i, gamma, duals, beta=1, t=1, mu=0, reg_sum=None, simplex=True):
    """ Modify n and duals!
    """

    w = n.compute_weights(t)
    g = n.sum_similarities * n.confidence * np.dot(n.margin.T, w) 

    if mu > 0:
        g -= mu*(n.alpha - reg_sum) 
    
    if simplex:
        # simplex constraint
        j = np.argmax(g)
        s_k = np.asarray([[1] if h==j else [0] for h in range(n.n)])
    else:
        # l1 constraint
        j = np.argmin(g)
        s_k = np.sign(g[j, :]) * beta * np.asarray([[1] if h==j else [0] for h in range(n.n)])

    alpha_k = (1 - gamma) * n.alpha + gamma * s_k
    n.set_alpha(alpha_k)

    # update duality gap
    duals[i] = (np.dot((s_k - alpha_k).squeeze(), g.squeeze()))

# --------------------------------------------------------------------- local learning

def local_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, simplex=True, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(nb_base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = float("inf")

    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, simplex))

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    return results

def regularized_local_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, mu=1, simplex=True, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(nb_base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = float("inf")
    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        reg_sum = [sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)]) for n in nodes]

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, 1, mu, reg_sum, simplex))

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    return results

def async_regularized_local_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, mu=1, simplex=True, callbacks=None):

    results = []
    N = len(nodes)

    # get margin matrices A
    for n in nodes:
        n.init_matrices(nb_base_clfs)
    
    # init with local classifiers
    duals = one_frank_wolfe_round(nodes, 1, beta, simplex)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = sum(duals)

    for t in range(nb_iter):

        gamma = 2 * N / (2 * N + t)

        # pick one node at random uniformally
        i = choice(range(len(nodes)))
        n = nodes[i]

        reg_sum = sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)])

        frank_wolfe_on_one_node(n, i, gamma, duals, beta, 1, mu, reg_sum, simplex)

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = sum(duals)

    return results

def neighbor_FW(nodes, nb_base_clfs=None, nb_iter=1, beta=1, simplex=True, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices()

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = float("inf")
    
    gamma = 1

    # frank-wolfe
    for t in range(nb_iter-1):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, simplex))

        for n in nodes:
            new_clfs = n.get_neighbors_clfs()
            n.set_margin_matrix(new_clfs)
            new_alpha = np.dot(n.clf, np.linalg.pinv(new_clfs)).T
            n.set_alpha(new_alpha)

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, simplex))

    results.append({})  
    for k, call in callbacks.items():
        results[t+2][k] = call[0](nodes, *call[1])
    results[t+2]["duality-gap"] = dual_gap

    return results

# ---------------------------------------------------------------- global consensus FW

def average_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, simplex=True, weighted=False, callbacks=None):

    results = []
    
    # get margin matrices A
    for n in nodes:
        n.init_matrices(nb_base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = float("inf")

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, simplex))

        # averaging between neighbors
        new_alphas = []
        for n in nodes:
            alphas = np.hstack([i.alpha for i in n.neighbors])

            if weighted:
                new_alphas.append(np.average(alphas, weights=n.sim, axis=1)[:, None])
            else:
                new_alphas.append(np.mean(alphas, axis=1)[:, None])

        for n, a in zip(nodes, new_alphas):
            n.set_alpha(a)

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    return results

def centralized_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, simplex=True, callbacks=None):

    results = []

    node = centralize_data(nodes)
    node.init_matrices(nb_base_clfs)

    list_node = [node]

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](list_node, *call[1])
    results[0]["duality-gap"] = float("inf")

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(list_node, gamma, beta, simplex))

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](list_node, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    final_alpha = list_node[0].alpha
    for n in nodes:
        n.init_matrices(nb_base_clfs)
        n.set_alpha(final_alpha)

    return results