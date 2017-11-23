import numpy as np

from network import centralize_data

"""
Boosting algorithms based on Frank Wolfe optimization
"""

# ----------------------------------------------------------- specific utils

def one_frank_wolfe_round(nodes, gamma, beta=1, t=1, simplex=True):
    """ Modify nodes!
    """

    for n in nodes:

        w = np.exp(-np.dot(n.margin, n.alpha)/t)
        w = np.nan_to_num(w/np.sum(w))

        # minimize negative gradient
        g = np.dot(n.margin.T, w)  
        
        if simplex:
            # simplex constraint
            j = np.argmax(g)
            s_k = np.asarray([[1] if i==j else [0] for i in range(n.n)])
        else:
            # l1 constraint
            j = np.argmin(g)
            s_k = np.sign(g[j, :]) * beta * np.asarray([[1] if i==j else [0] for i in range(n.n)])

        n.set_alpha((1 - gamma) * n.alpha + gamma * s_k)

# --------------------------------------------------------------------- local learning

def local_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, simplex=True, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(nb_base_clfs)
    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (3 + t)

        one_frank_wolfe_round(nodes, gamma, beta, simplex)

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call[0](nodes, *call[1])

    return results

def neighbor_FW(nodes, nb_base_clfs=None, nb_iter=1, beta=1, simplex=True, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices()
    
    # frank-wolfe
    for t in range(nb_iter-1):

        gamma = 2 / (3 + t)

        one_frank_wolfe_round(nodes, gamma, beta, simplex)

        for n in nodes:
            new_clfs = n.get_neighbors_clfs()
            n.set_margin_matrix(new_clfs)
            new_alpha = np.dot(n.clf, np.linalg.pinv(new_clfs)).T
            n.set_alpha(new_alpha)

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call[0](nodes, *call[1])

    one_frank_wolfe_round(nodes, gamma, beta, simplex)

    results.append({})  
    for k, call in callbacks.items():
        results[t+1][k] = call[0](nodes, *call[1])

    return results

# ---------------------------------------------------------------- global consensus FW

def average_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, simplex=True, callbacks=None):

    results = []
    
    # get margin matrices A
    for n in nodes:
        n.init_matrices(nb_base_clfs)

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (3 + t)

        one_frank_wolfe_round(nodes, gamma, beta, simplex)

        # averaging between neighbors
        for n in nodes:

            n.set_alpha((n.alpha + sum([i.alpha for i in n.neighbors]))/(1 + len(n.neighbors)))

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call[0](nodes, *call[1])

    return results

def centralized_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, simplex=True, callbacks=None):

    results = []

    node = centralize_data(nodes)
    node.init_matrices(nb_base_clfs)

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (3 + t)

        one_frank_wolfe_round([node], gamma, beta, simplex)

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call[0]([node], *call[1])

    return results
