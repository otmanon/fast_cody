import scipy as sp
import numpy as np

'''
projects linear space to be orthognal to J using constrained least squares
B - 3n x m subspace matrix
J - c x 3n desired null space of B

Optional
M - 3n x 3n mass matrix

returns
B - 3n x m subspace matrix s.t. B^T M J = 0
'''
def project_to_orthogonal(B, J, M=None):
    if M is None:
        M = sp.sparse.identity(B.shape[0])
    # M = sp.sparse.identity(B.shape[0])

    Mi = sp.sparse.diags(1 / M.diagonal(), 0)

    JMi = J @ Mi
    JMiJ = JMi @ J.T

    f = J @ B
    g = np.linalg.solve(JMiJ, f)
    MB = M @ B
    B2 = Mi @ (- J.T @ g  + MB)
    return B2

