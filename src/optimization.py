import numpy as np
from random import randint

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from classification import get_double_basis
from evaluation import losses
from network import centralize_data, set_edges, get_alphas
from utils import square_root_matrix, get_adj_matrix, stack_results, kalo_utils

"""
Boosting algorithms using Frank Wolfe optimization
"""

# ----------------------------------------------------------- specific utils

def graph_discovery(nodes, similarities, k=1, *args):
    
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
    np.fill_diagonal(res, 0.)

    return res

def obj_kalo(w, z, S, l, mu, la):

    d = S.dot(w)

    # print("kalo", d.dot(l), mu * w.dot(z) / 2, - np.log(d).sum(), b * w.dot(w))
    if np.count_nonzero(d) < len(d):
        return np.inf

    return d.dot(l) + (mu / 2) * (w.dot(z) - np.log(d).sum() + la * (mu / 2) * w.dot(w))

def kalo_graph_discovery(nodes, similarities, S, triu_ix, map_idx, mu=1, la=1, *args):

    n = len(nodes)
    n_pairs = n * (n - 1) // 2
    stop_thresh = 10e-2 / n_pairs

    z = pairwise_distances(np.hstack(get_alphas(nodes)).T)**2
    z = z[triu_ix]

    l = np.asarray(losses(nodes))

    if similarities is not None:
        w = np.asarray(similarities[triu_ix])
    else:
        w = np.ones(n_pairs)
    d = S.dot(w)

    gamma = 1 / (np.linalg.norm(l.dot(S)) + (mu / 2) * (np.linalg.norm(z) + np.linalg.norm(S.T.dot(S)) + 2 * la * (mu / 2)))
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
    similarities = np.zeros((n, n))
    similarities[triu_ix] = similarities.T[triu_ix] = w

    return similarities

def block_kalo_graph_discovery(nodes, similarities, S, triu_ix, map_idx, mu=1, la=1, kappa=1, *args):

    n = len(nodes)
    n_pairs = n * (n - 1) // 2
    stop_thresh = 10e-6

    z = pairwise_distances(np.hstack(get_alphas(nodes)).T)**2
    z = z[triu_ix]

    l = np.asarray(losses(nodes))

    if similarities is not None:
        w = np.asarray(similarities[triu_ix])
    else:
        w = np.ones(n_pairs)
    d = S.dot(w)

    gamma = 1 / (np.linalg.norm(l.dot(S)) + (mu / 2) * (np.linalg.norm(z) + np.linalg.norm(S.T.dot(S)) + 2 * la * (mu / 2)))
    obj = obj_kalo(w, z, S, l, mu, la)

    w, new_w = np.ones(n_pairs), np.ones(n_pairs)

    grad = np.zeros(kappa)
    print('\n', "it=", 0, "obj=", obj, "gamma=", gamma)
    for k in range(2000):

        rnd_j = np.random.choice(n, 1+kappa)
        i, others = rnd_j[0], rnd_j[1:]
        ides = []

        for e, j in enumerate(others):

            idx = map_idx[min(i, j), max(i, j)]
            grad[e] = l[i] + l[j] + (mu / 2) * (z[idx] - (1 / d[i] + 1 / d[j]) + 2 * la * (mu / 2) * w[idx])
            ides.append(idx)

        new_w[ides] = w[ides] - gamma * grad
        new_w[new_w < 0] = 0

        new_obj = obj_kalo(new_w, z, S, l, mu, la)

        # print("new obj=", new_obj)

        if new_obj > obj and (gamma / 2) > 0:
            gamma /= 2

        elif abs(obj - new_obj) > abs(stop_thresh * obj):
            obj = new_obj
            w = new_w
            gamma *= 1.05
            # print(k, obj)

        else:
            w = new_w
            break
        
        d = S.dot(w)

    print(k, new_obj)

    # print("done in", k)
    similarities = np.zeros((n, n))
    similarities[triu_ix] = similarities.T[triu_ix] = w

    return similarities

gd_func_dict = {
    "block_kalo": block_kalo_graph_discovery,
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

def gd_reg_local_FW(nodes, base_clfs, gd_method={"name":"uniform", "pace_gd":1, "args":()}, nb_iter=1, beta=None, mu=1, monitors=None, checkevery=1):

    results = []
    N = len(nodes)

    if gd_method["name"] in ["kalo", "block_kalo"]:

        S, triu_ix, map_idx = kalo_utils(N)
        gd_method["args"] = (S, triu_ix, map_idx,) + gd_method["args"]

    gd_function = gd_func_dict[gd_method["name"]]
    gd_args = gd_method["args"]
    gd_pace = gd_method["pace_gd"]

    # iterations = [0] * N

    # start from local models
    local_FW(nodes, base_clfs, beta=beta, nb_iter=nb_iter, monitors={})

    # init graph
    similarities = gd_function(nodes, None, *gd_args)
    adj_matrix = get_adj_matrix(similarities, 1e-3)

    # get margin matrices A and reinit local models and graph
    for n in nodes:
        n.init_matrices(base_clfs)
    set_edges(nodes, similarities, adj_matrix)

    stack_results(nodes, results, 0, monitors)

    duals = [0] * N

    for t in range(1, nb_iter+1):

        if t % gd_pace == 0:

            # graph discovery
            similarities = gd_function(nodes, similarities, *gd_args)
            adj_matrix = get_adj_matrix(similarities, 1e-3)
            set_edges(nodes, similarities, adj_matrix)

        # pick one node at random uniformly
        i = randint(0, len(nodes)-1)
        n = nodes[i]

        gamma = 2*N / (2*N + t)
        # iterations[i] += 1

        reg_sum = sum([s*m.alpha for m, s in zip(n.neighbors, n.sim)])

        frank_wolfe_on_one_node(n, i, gamma, duals, beta, 1, mu, reg_sum)

        dual_gap = sum(duals)

        if t % checkevery == 0:
            stack_results(nodes, results, dual_gap, monitors, similarities)

    results[-1]["adj-matrix"] = adj_matrix
    results[-1]["similarities"] = similarities

    return results

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
