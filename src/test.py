import cvxpy as cvx
import numpy as np

N = 4
k = 2

alpha = np.asarray([[1,0], [1,1], [0.5,1], [2,2]]).T

x = cvx.Variable(N, N)

# set node degrees
degree_matrix = np.eye(N)

objective = cvx.Minimize(cvx.trace(alpha * (degree_matrix - x) * alpha.T))
constraints = [x >= np.zeros((N,N)), x < np.ones((N,N))/k, cvx.trace(x) == 0., cvx.sum_entries(x, 1) == np.ones(N), cvx.sum_entries(x, 0) == np.ones((1, N))]

prob = cvx.Problem(objective, constraints)
result = prob.solve()

res = np.asarray(x.value)
assert np.allclose(res, res.T)

# drop insignificant edges
res[res < 1/N] = 0.

print(res)