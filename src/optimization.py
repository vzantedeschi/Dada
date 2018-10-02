import cvxpy as cvx
import numpy as np
from random import randint

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from classification import get_double_basis
from evaluation import losses
from kalofolias import obj_kalo
from network import centralize_data, set_edges, get_alphas
from utils import square_root_matrix, get_adj_matrix, stack_results

"""
Boosting algorithms using Frank Wolfe optimization
"""

# ----------------------------------------------------------- specific utils

def graph_discovery(nodes, k=1, *args):
    
    N = len(nodes)

    alpha = np.hstack([n.alpha for n in nodes])
    x = cvx.Variable(N, N)

    # set node degrees
    degree_matrix = np.eye(N)
    
    objective = cvx.Minimize(cvx.trace(alpha * (degree_matrix - x) * alpha.T))
    constraints = [x >= np.zeros((N,N)), x < np.ones((N,N))/k, cvx.trace(x) == 0., cvx.sum_entries(x, 1) == np.ones(N), cvx.sum_entries(x, 0) == np.ones((1, N))]

    prob = cvx.Problem(objective, constraints)
    result = prob.solve()

    res = np.asarray(x.value)
    # assert np.allclose(res, res.T)

    # drop insignificant edges
    res[res < 1/N] = 0.

    return res

def kalo_graph_discovery(nodes, mu=1, b=1, *args):

    stop_thresh = 0.001

    n = len(nodes)
    n_pairs = int(n * (n - 1) / 2)

    z = pairwise_distances(np.hstack(get_alphas(nodes)).T)**2
    z = z[np.triu_indices(n, 1)]

    l = np.asarray(losses(nodes))

    # construct mapping matrix from 2D index to 1D index for convenience
    map_idx = np.ones((n, n), dtype=int)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            map_idx[i, j] = k
            k += 1

    # Construct linear transformation matrix mapping weight vector to degree vector
    S = np.zeros((n, n_pairs))
    for i in range(n):
        for j in range(n):
            if j != i:
                # S[i, map_idx[min(i, j), max(i, j)]] = 1 / nodes[i].confidence
                S[i, map_idx[min(i, j), max(i, j)]] = 1


    w = np.ones(n_pairs)
    similarities = np.zeros((n, n))
    d = S.dot(w)

    gamma = 1 / (np.linalg.norm(l.dot(S)) + mu * np.linalg.norm(z) / 2 + np.linalg.norm(S.T.dot(S)) + 2 * b)
    obj = obj_kalo(w, z, S, l, mu, b)
    # print("objective 0", obj)
    k = 0
    while True:
        grad = l.dot(S) + mu * z / 2 - (1. / d).dot(S) + 2 * b * w
        # print(grad)
        new_w = w - gamma * grad
        new_w[new_w < 0] = 0

        k += 1

        if k % 100 == 0:
            new_obj = obj_kalo(new_w, z, S, l, mu, b)

            if np.isinf(new_obj):
                gamma *= 0.1 
                # print("inf")

            elif abs(obj - new_obj) > abs(stop_thresh * obj):
                obj = new_obj
                w = new_w
                # print("continue")

            else:
                # print("break")
                w = new_w
                break
        
        d = S.dot(w)

    # print("objective %d" % k, new_obj)

    i, j = 0, 1
    for k in w: 
        similarities[j, i] = k
        similarities[i, j] = k
        j += 1
        if j == n:
            i += 1
            j = i + 1 

    return similarities

gd_func_dict = {
    "kalo": kalo_graph_discovery,
    "uniform": graph_discovery
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

    if mu > 0 and type(reg_sum) != int:
        g += mu*(n.sum_similarities * n.alpha - reg_sum) 

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

def local_FW(nodes, base_clfs, nb_iter=1, beta=None, monitors=None, checkevery=1):

    results = []
    
    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    stack_results(nodes, results, 0, monitors)

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta))

        if t % checkevery == 0:
            stack_results(nodes, results, dual_gap, monitors)

    return results

def global_regularized_local_FW(nodes, base_clfs, nb_iter=1, beta=None, monitors=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)
    alpha0 = np.zeros((len(base_clfs), 1))

    stack_results(nodes, results, 0, monitors)
    
    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap, alpha0 = global_reg_frank_wolfe(nodes, gamma, alpha0, beta=beta, t=1)

        stack_results(nodes, results, dual_gap, monitors)

    return results

# def regularized_local_FW(nodes, base_clfs, nb_iter=1, beta=None, mu=1, monitors=None):

#     results = []

#     # get margin matrices A
#     for n in nodes:
#         n.init_matrices(base_clfs)

#     results.append({})  
#     for k, call in monitors.items():
#         results[0][k] = call[0](nodes, *call[1])
#     results[0]["duality-gap"] = 0
    
#     # frank-wolfe
#     for t in range(nb_iter):

#         gamma = 2 / (2 + t)

#         reg_sum = [sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)]) for n in nodes]

#         dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, 1, mu, reg_sum))

#         results.append({})  
#         for k, call in monitors.items():
#             results[t+1][k] = call[0](nodes, *call[1])
#         results[t+1]["duality-gap"] = dual_gap

#     return results

def regularized_local_FW(nodes, base_clfs, nb_iter=1, beta=None, mu=1, monitors=None, checkevery=1):

    results = []
    N = len(nodes)

    iterations = [0] * N

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    stack_results(nodes, results, 0, monitors)

    duals = [0] * N

    for t in range(nb_iter):

        # pick one node at random uniformally
        i = randint(0, len(nodes)-1)
        n = nodes[i]

        gamma = 2 * N / (2 * N + t)
        iterations[i] += 1

        reg_sum = sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)])

        frank_wolfe_on_one_node(n, i, gamma, duals, beta, 1, mu, reg_sum)

        if t % checkevery == 0:
            stack_results(nodes, results, sum(duals), monitors)

    return results

def gd_reg_local_FW(nodes, base_clfs, init_w, gd_method={"name":"uniform", "pace_gd":1, "args":()}, nb_iter=1, beta=None, mu=1, reset_step=False, monitors=None, checkevery=1):

    results = []
    N = len(nodes)

    gd_function = gd_func_dict[gd_method["name"]]
    gd_args = gd_method["args"]
    gd_pace = gd_method["pace_gd"]

    iterations = [0] * N

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs, n.alpha)
        
    adj_matrix = get_adj_matrix(init_w, 1e-3)
    set_edges(nodes, init_w, adj_matrix)

    stack_results(nodes, results, 0, monitors)

    duals = [0] * N

    resettable_t = 0
    for t in range(nb_iter):

        # pick one node at random uniformally
        i = randint(0, len(nodes)-1)
        n = nodes[i]

        gamma = 2*N / (2*N + t)
        iterations[i] += 1

        reg_sum = sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)])

        frank_wolfe_on_one_node(n, i, gamma, duals, beta, 1, mu, reg_sum)

        dual_gap = sum(duals)

        if t % checkevery == 0:
            stack_results(nodes, results, dual_gap, monitors)

        if resettable_t % gd_pace == 0:

            # graph discovery
            similarities = gd_function(nodes, *gd_args)
            adj_matrix = get_adj_matrix(similarities, 1e-3)
            set_edges(nodes, similarities, adj_matrix)

            if reset_step:
                resettable_t = 0

    results[-1]["adj-matrix"] = adj_matrix
    results[-1]["similarities"] = similarities

        resettable_t += 1

    return results

# def gd_reg_local_FW(nodes, base_clfs, local_alphas, gd_method={"name":"uniform", "pace_gd":1, "args":()}, nb_iter=1, beta=None, mu=1, reset_step=False, monitors=None):

#     N = len(nodes)
#     results = []

#     gd_function = gd_func_dict[gd_method["name"]]
#     gd_args = gd_method["args"]
#     gd_pace = gd_method["pace_gd"]

#     # get margin matrices A
#     for n, alpha in zip(nodes, local_alphas):
#         n.init_matrices(base_clfs, alpha=alpha)
#     set_edges(nodes, np.eye(len(nodes)), np.eye(len(nodes)))

#     results.append({})  
#     for k, call in monitors.items():
#         results[0][k] = call[0](nodes, *call[1])
#     results[0]["duality-gap"] = 0

#     resettable_t = 0
#     # frank-wolfe
#     for t in range(nb_iter):

#         gamma = 2 / (2 + resettable_t)

#         reg_sum = [sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)]) for n in nodes]

#         dual_gap = sum(one_frank_wolfe_round(nodes, gamma, beta, 1, mu, reg_sum))

#         results.append({})  
#         for k, call in monitors.items():
#             results[t+1][k] = call[0](nodes, *call[1])
#         results[t+1]["duality-gap"] = dual_gap

#         resettable_t += 1

#         if resettable_t % gd_pace == 0:

#             # graph discovery
#             similarities = gd_function(nodes, gd_args)
#             adj_matrix = get_adj_matrix(similarities, 1e-3)
#             set_edges(nodes, similarities, adj_matrix)

#             if reset_step:
#                 resettable_t = 0

#             results[t+1]["adj-matrix"] = similarities

#     return results
# ---------------------------------------------------------------- global consensus FW

def average_FW(nodes, base_clfs, nb_iter=1, beta=None, weighted=False, monitors=None):

    results = []

    # get margin matrices A
    for n in nodes:
        n.init_matrices(base_clfs)

    stack_results(nodes, results, 0, monitors)

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

        stack_results(nodes, results, dual_gap, monitors)

    return results

def centralized_FW(nodes, base_clfs, nb_iter=1, beta=None, monitors=None, checkevery=1):

    results = []

    node = centralize_data(nodes)
    node.init_matrices(base_clfs)

    list_node = [node]

    stack_results(list_node, results, 0, monitors)

    # frank-wolfe
    for t in range(nb_iter):

        gamma = 2 / (2 + t)

        dual_gap = sum(one_frank_wolfe_round(list_node, gamma, beta))

        if t % checkevery == 0:
            stack_results(list_node, results, dual_gap, monitors)

    final_alpha = list_node[0].alpha
    for n in nodes:
        n.init_matrices(base_clfs)
        n.set_alpha(final_alpha)

    return results
