import numpy as np

# ----------------------------------------------------------- specifid utils

def one_frank_wolfe_round(t, nodes, beta):
    """ Modify nodes!
    """

    gamma = 2 / (3 + t)

    for n in nodes:

        w = np.exp(-np.dot(n.margin, n.alpha))
        w = np.nan_to_num(w/np.sum(w))

        # minimize negative gradient
        g = np.dot(w.T, n.margin)    
        j = np.argmin(g)

        s_k = np.sign(g[:, j]) * beta * n.base_clfs[:, j][:, np.newaxis]
        n.set_alpha((1 - gamma) * n.alpha + gamma * s_k)

# --------------------------------------------------------------------- local learning

def local_FW(nodes, nb_iter=1, beta=1, callbacks=None):

    results = []
    
    # frank-wolfe
    for t in range(nb_iter):

        one_frank_wolfe_round(t, nodes, beta)

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call(nodes)

    return results

# def neighbor_FW(nodes, nb_iter=1, beta=1, callbacks=None):

#     results = []
    
#     # frank-wolfe
#     for t in range(nb_iter):

#         one_frank_wolfe_round(t, nodes, beta)

#         if t < nb_iter-1:
#             for n in nodes:
#                 new_clfs = n.get_neighbors_clfs()
#                 n.set_margin_matrix(new_clfs)
#                 new_alpha = np.dot(n.clf, np.linalg.pinv(new_clfs)).T
#                 n.set_alpha(new_alpha)

#         results.append({})  
#         for k, call in callbacks.items():
#             results[t][k] = call(nodes)

#     return results

# ---------------------------------------------------------------- global consensus FW

def average_FW(nodes, nb_iter=1, beta=1, callbacks=None):

    results = []
    
    # frank-wolfe
    for t in range(nb_iter):

        one_frank_wolfe_round(t, nodes, beta)

        # averaging between neighbors
        for n in nodes:

            n.set_alpha((n.alpha + sum([i.alpha for i in n.neighbors]))/(1 + len(n.neighbors)))

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call(nodes)

    return results
