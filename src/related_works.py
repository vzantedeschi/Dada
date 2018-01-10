from copy import deepcopy
from math import log

import numpy as np

# ----------------------------------------------------------------------- Related Work

def minimize_gradients(gradients, nodes, gamma, beta=1, simplex=True):

    new_alphas = []
    for n, g in zip(nodes, gradients):

        if simplex:
            # simplex constraint
            j = np.argmax(g)
            s_k = np.asarray([[1] if i==j else [0] for i in range(n.n)])
        else:
            # l1 constraint
            j = np.argmin(g)
            s_k = np.sign(g[j, :]) * beta * np.asarray([[1] if i==j else [0] for i in range(n.n)])

        new_alphas.append((1 - gamma) * n.alpha + gamma * s_k)

    return new_alphas

def gac_routine(vectors, nodes, nb_iter):

    # update gradients using GAC routine
    new_vectors = []
    
    for _ in range(nb_iter):

        new_vectors = []

        for n in nodes:

            new_vectors.append(np.sum([s*vectors[j] for j, (_, s) in enumerate(zip(n.neighbors, n.sim))], axis=0))

        vectors = deepcopy(new_vectors)

    return new_vectors

def lafond_FW(nodes, nb_base_clfs, nb_iter=1, beta=1, c1=5, t=1, simplex=True, callbacks=None):

    results = []
    
    # get margin matrices A
    for n in nodes:
        n.init_matrices(nb_base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])

    # frank-wolfe
    for i in range(nb_iter):

        nb_iter_gac = int(c1 + log(i+1))

        gamma = 2 / (2 + i)

        # compute init gradients
        gradients = []

        for n in nodes:

            w = n.compute_weights(i)
            g = np.dot(n.margin.T, w)
            
            gradients.append(g)

        gradients = gac_routine(gradients, nodes, nb_iter_gac)

        alphas = minimize_gradients(gradients, nodes, gamma, beta, simplex)

        alphas = gac_routine(alphas, nodes, nb_iter_gac)

        for n, a in zip(nodes, alphas):
            n.set_alpha(a)

        results.append({})  
        for k, call in callbacks.items():
            results[i+1][k] = call[0](nodes, *call[1])

    return results