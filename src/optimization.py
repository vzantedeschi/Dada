import numpy as np

from evaluation import alpha_variance
# ---------------------------------------------------------------------------- global consensus FW

def average_FW(nodes, nb_iter=1):
    
    # frank-wolfe
    for t in range(1, nb_iter+1):

        gamma = 2 / (2 + t)

        for n in nodes:

            w = np.exp(-np.dot(n.margin, n.alpha))
            w /= np.sum(w)

            j = np.argmax(np.dot(w.T, n.margin))
            n.set_alpha((1 - gamma) * n.alpha + gamma * n.base_clfs[:, j][:, np.newaxis])

        # averaging 
        for n in nodes:

            n.set_alpha((n.alpha + sum([nodes[i].alpha for i in n.neighbors]))/(1 + len(n.neighbors)))

        print(alpha_variance(nodes))
