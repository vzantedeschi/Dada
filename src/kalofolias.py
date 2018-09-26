"""test lp graph."""

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances


def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 
    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project
    s: int, optional, default: 1,
       radius of the simplex
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def obj_kalo(w, S, z, alpha, beta):
    if np.any(w < 0):
        return np.inf
    else:
        return 2 * w.dot(z) - alpha * np.log(S.dot(w)).sum() + beta * w.dot(w)


def obj_dong(w, S, z, alpha, c):
    if np.any(w < 0) or not np.isclose(w.sum(), c):
        return np.inf
    else:
        return 2 * w.dot(z) + alpha * (2 * w.dot(w) + (S.dot(w)**2).sum())

if __name__ == '__main__':
    ################ SETUP PROBLEM ###############

    n = 5
    dim = 10
    n_pairs = int(n * (n - 1) / 2)

    np.random.seed(42)
    # alpha = np.random.rand(n, dim)
    alpha = [[0, 0], [0, 1], [1, 0], [1.1, 0], [1.2, 0]]
    z = pairwise_distances(alpha)**2
    z = z[np.triu_indices(n, 1)]

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
                S[i, map_idx[min(i, j), max(i, j)]] = 1

    alpha = 1
    beta = 1

    ############## KALOFOLIAS MODEL ################

    ## GRADIENT DESCENT ALGORITHM

    # gamma = 0.01
    gamma = 1 / (2 * np.linalg.norm(z) + alpha * np.linalg.norm(S.T.dot(S)) +
                 2 * beta)
    print("step:", gamma)

    w = np.ones(n_pairs)
    obj = obj_kalo(w, S, z, alpha, beta)

    k = 0
    while True:

        d = S.dot(w)
        grad = 2 * z - alpha * (1. / d).dot(S) + 2 * beta * w

        w = w - gamma * grad
        w[w < 0] = 0
        k += 1

        if k % 100 == 0:
            new_obj = obj_kalo(w, S, z, alpha, beta)
            print(new_obj)
            if abs(obj - new_obj) > abs(0.00001 * obj):
                obj = new_obj
            else:
                break

    print(k)
    print(z)
    print(w[:10])
    print((w > 0).sum() / n_pairs)

    ## COORDINATE DESCENT ALGORITHM

    # w = np.ones(n_pairs)

    # gamma = 0.1
    # for k in range(100000):
    #     if k % 1000 == 0:
    #         print(k, obj_kalo(w, S, z, alpha, beta))
    #     i, j = np.random.choice(n, 2)
    #     idx = map_idx[min(i, j), max(i, j)]
    #     d = S.dot(w)
    #     grad = 2 * z[idx] - alpha * (1 / d[i] + 1 / d[j]) + 2 * beta * w[idx]
    #     w[idx] = w[idx] - gamma * grad
    #     w[w < 0] = 0

    # print(z)
    # print(w[:10])
    # print((w > 0).sum() / n_pairs)

    ############## DONG MODEL ################

    ## GRADIENT DESCENT ALGORITHM

    alpha = 1
    c = 1

    gamma = .01
    # gamma = 1 / (alpha * (4 + 2 * np.linalg.norm(S.T.dot(S))**2))
    print("step:", gamma)

    w = np.ones(n_pairs) / n_pairs

    for k in range(1000):
        if k % 100 == 0:
            print(k, obj_dong(w, S, z, alpha, c))
        d = S.dot(w)
        grad = 2 * z + alpha * (4 * w + 2 * S.dot(w).T.dot(S))
        w = w - gamma * grad
        w = euclidean_proj_simplex(w, c)

    print(z)
    print(w[:10])
    print((w > 0).sum() / n_pairs)
