import numpy as np

# --------------------------------------------------------------------- local learning

def local_FW(nodes, nb_iter=1, beta=1, callbacks=None):

    results = []
    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (3 + t)

        for n in nodes:

            w = np.exp(-np.dot(n.margin, n.alpha))
            w /= np.sum(w)

            # minimize negative gradient
            g = np.dot(w.T, n.margin)    
            j = np.argmin(g)

            s_k = np.sign(g[:, j]) * beta * n.base_clfs[:, j][:, np.newaxis]
            n.set_alpha((1 - gamma) * n.alpha + gamma * s_k)

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call(nodes)

    return results

# ---------------------------------------------------------------- global consensus FW

def average_FW(nodes, nb_iter=1, beta=1, callbacks=None):

    results = []
    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (3 + t)

        for n in nodes:

            w = np.exp(-np.dot(n.margin, n.alpha))
            w /= np.sum(w)

            # minimize negative gradient
            g = np.dot(w.T, n.margin)    
            j = np.argmin(g)

            s_k = np.sign(g[:, j]) * beta * n.base_clfs[:, j][:, np.newaxis]
            n.set_alpha((1 - gamma) * n.alpha + gamma * s_k)

        # averaging between neighbors
        for n in nodes:

            n.set_alpha((n.alpha + sum([nodes[i].alpha for i in n.neighbors]))/(1 + len(n.neighbors)))

        results.append({})  
        for k, call in callbacks.items():
            results[t][k] = call(nodes)

    return results
