import cvxpy as cvx
import numpy as np
from random import choice

from sklearn.neighbors import NearestNeighbors

from classification import get_double_basis
from network import centralize_data, set_edges
from utils import square_root_matrix

"""
Boosting algorithms using Frank Wolfe optimization
"""

# ----------------------------------------------------------- specific utils

def graph_discovery(nodes):

    alpha = np.hstack([n.alpha for n in nodes])

    laplacian = square_root_matrix(np.dot(alpha.T, alpha))

    laplacian /= np.trace(laplacian)

    return (np.eye(len(nodes))-laplacian).clip(min=0)

def graph_discovery_sparse(nodes, *args):
    
    N = len(nodes)

    alpha = np.hstack([n.alpha for n in nodes])

    x = cvx.Variable(N, N)

    # set node degrees to 1
    objective = cvx.Minimize(cvx.trace(alpha * (np.eye(N) - x) * alpha.T))
    constraints = [x > np.zeros((N,N)), cvx.trace(x) == 0, cvx.norm(x, 1) <= N, cvx.sum_entries(x, axis=1) == np.ones(N), cvx.sum_entries(x, axis=0) == np.ones((1,N))]

    prob = cvx.Problem(objective, constraints)
    result = prob.solve()

    res = np.asarray(x.value)

    return res.clip(min=0)

def graph_discovery_knn(nodes, k=10):

    N = len(nodes)

    alpha = np.hstack([n.alpha for n in nodes])

    x = cvx.Variable(N, N)

    # set node degrees to 1
    objective = cvx.Minimize(cvx.trace(alpha * (np.eye(N) - x) * alpha.T))
    constraints = [x > np.zeros((N,N)), cvx.trace(x) == 0, cvx.sum_entries(x, axis=1) == np.ones(N), cvx.sum_entries(x, axis=0) == np.ones((1,N))]

    prob = cvx.Problem(objective, constraints)
    result = prob.solve()   

    graph_sim = np.asarray(x.value).clip(min=0)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(1 - graph_sim)

    sparsity_mask = nbrs.kneighbors_graph(1 - graph_sim).toarray()

    # make it symmetric
    sparsity_mask = np.logical_and(sparsity_mask, sparsity_mask.T)

    return np.multiply(graph_sim, sparsity_mask)

def graph_discovery_full_knn(nodes, k=10):

    N = len(nodes)

    alpha = np.hstack([n.alpha for n in nodes])

    graph_sim = np.dot(alpha.T, alpha)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(1 - graph_sim)

    sparsity_mask = nbrs.kneighbors_graph(1 - graph_sim).toarray()

    # make it symmetric
    sparsity_mask = np.logical_and(sparsity_mask, sparsity_mask.T)

    return np.multiply(graph_sim, sparsity_mask)

gd_func_dict = {
    "laplacian": graph_discovery_sparse,
    "knn": graph_discovery_knn,
    "full-knn": graph_discovery_full_knn,
}

def one_frank_wolfe_round(nodes, gamma, beta=None, t=1, mu=0, reg_sum=None):
    """ Modify nodes!
    """
 
    duals = [0] * len(nodes)

    for i, n in enumerate(nodes):

        if reg_sum:
            r = reg_sum[i]
        else:
            r = None

        frank_wolfe_on_one_node(n, i, gamma, duals, beta, t, mu, r)

    return duals

def frank_wolfe_on_one_node(n, i, gamma, duals, beta=None, t=1, mu=0, reg_sum=None):
    """ Modify n and duals!
    """

    w = n.compute_weights(t)
    g = - n.sum_similarities * n.confidence * np.dot(n.margin.T, w) 

    if mu > 0:
        g += mu*(n.alpha - reg_sum) 

    if beta is None:
        # simplex constraint
        j = np.argmin(g)
        s_k = np.asarray([[1] if h==j else [0] for h in range(n.n)])
    else:
        # l1 constraint
        j = np.argmax(abs(g))
        s_k = np.sign(-g[j, :]) * beta * np.asarray([[1] if h==j else [0] for h in range(n.n)])

    alpha_k = (1 - gamma) * n.alpha + gamma * s_k
    n.set_alpha(alpha_k)

    # update duality gap
    duals[i] = (np.dot((alpha_k - s_k).squeeze(), g.squeeze()))

def global_reg_frank_wolfe(nodes, gamma, alpha0, beta=None, t=1):
    """ Modify n and duals!
    """
    K = len(nodes)
    gradients = []
    alphas = []

    for i, n in enumerate(nodes):

        w = n.compute_weights(t)
        gradients.append( - n.sum_similarities * n.confidence * np.dot(n.margin.T, w))
        alphas.append(n.alpha)
    
    gradients.append(np.sum(gradients, axis=0))
    g = np.vstack(gradients)
    # print("g=", g, "\n")

    if beta is None:
        # simplex constraint
        j = np.argmin(g)
        s = np.asarray([[1] if h==j else [0] for h in range(n.n*(K+1))])
    else:
        # l1 constraint
        j = np.argmax(abs(g))
        s = np.sign(-g[j, :]) * beta * np.asarray([[1] if h==j else [0] for h in range(n.n*(K+1))])

    # retreive vector to update
    i = j // n.n # node
    j = j % n.n # coordinate

    if beta is None:
        s_i = np.asarray([[1] if h==j else [0] for h in range(n.n)])
    else:
        s_i = np.sign(-g[j, :]) * beta * np.asarray([[1] if h==j else [0] for h in range(n.n)])

    if i == K:
        alpha0 = (1 - gamma) * alpha0 + gamma * s_i

        for n in nodes:
            n.set_alpha(alpha0=alpha0)
    else:
        alpha = (1 - gamma) * nodes[i].alpha + gamma * s_i
        nodes[i].set_alpha(alpha)
        alphas[i] = alpha        
        # print(alpha)
    alphas.append(alpha0)

    # update duality gap
    dual = (np.dot((np.vstack(alphas) - s).squeeze(), g.squeeze()))

    return dual, alpha0

# --------------------------------------------------------------------- local learning

def local_FW(nodes, base_clfs, nb_iter=1, beta=None, callbacks=None):

    results = []
    
    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = 0

    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta))

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    return results

def global_regularized_local_FW(nodes, base_clfs, nb_iter=1, beta=None, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)
    alpha0 = np.zeros((len(base_clfs), 1))

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = 0
    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap, alpha0 = global_reg_frank_wolfe(nodes, gamma, alpha0, beta=beta, t=1)

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    return results

def regularized_local_FW(nodes, base_clfs, nb_iter=1, beta=None, mu=1, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = 0
    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        reg_sum = [sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)]) for n in nodes]

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, 1, mu, reg_sum))

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    return results

def async_regularized_local_FW(nodes, base_clfs, nb_iter=1, beta=None, mu=1, callbacks=None):

    results = []
    N = len(nodes)

    iterations = [0] * N

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = 0

    duals = [0] * N

    for t in range(nb_iter):

        # pick one node at random uniformally
        i = choice(range(len(nodes)))
        n = nodes[i]

        gamma = 2 * N / (2 * N + iterations[i])
        iterations[i] += 1

        reg_sum = sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)])

        frank_wolfe_on_one_node(n, i, gamma, duals, beta, 1, mu, reg_sum)

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = sum(duals)

    return results

def gd_reg_local_FW(nodes, base_clfs, gd_method={"name":"laplacian", "pace_gd":1, "args":()}, nb_iter=1, beta=None, mu=1, reset_step=True, callbacks=None):

    N = len(nodes)
    results = []

    gd_function = gd_func_dict[gd_method["name"]]
    gd_args = gd_method["args"]
    gd_pace = gd_method["pace_gd"]

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)
    set_edges(nodes, np.eye(len(nodes)))

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = 0

    resettable_t = 0
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + resettable_t)

        reg_sum = [sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)]) for n in nodes]

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, 1, mu, reg_sum))

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](nodes, *call[1])
        results[t+1]["duality-gap"] = dual_gap

        resettable_t += 1

        if resettable_t % gd_pace == 0 and dual_gap < N:

            # graph discovery
            adj_matrix = gd_function(nodes, gd_args)
            set_edges(nodes, adj_matrix)

            if reset_step:
                resettable_t = 0

            results[t+1]["adj-matrix"] = adj_matrix

    return results
# ---------------------------------------------------------------- global consensus FW

def average_FW(nodes, base_clfs, nb_iter=1, beta=None, weighted=False, callbacks=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](nodes, *call[1])
    results[0]["duality-gap"] = 0

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta))

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

def centralized_FW(nodes, base_clfs, nb_iter=1, beta=None, callbacks=None):

    results = []

    node = centralize_data(nodes)
    node.init_matrices(base_clfs)

    list_node = [node]

    results.append({})  
    for k, call in callbacks.items():
        results[0][k] = call[0](list_node, *call[1])
    results[0]["duality-gap"] = 0

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(list_node, gamma, beta))

        results.append({})  
        for k, call in callbacks.items():
            results[t+1][k] = call[0](list_node, *call[1])
        results[t+1]["duality-gap"] = dual_gap

    final_alpha = list_node[0].alpha
    for n in nodes:
        n.init_matrices(base_clfs)
        n.set_alpha(final_alpha)

    return results
