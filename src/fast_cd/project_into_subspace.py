import scipy as sp
import numpy as np
'''
projects x into a subspace B via least squares
'''
def project_into_subspace(x, B, M=None):
    if M is None:
        M = sp.sparse.identity(x.shape[0])
    if M.shape[0] == B.shape[0]//3:
        M = sp.sparse.kron(sp.sparse.identity(3), M)
    BM = B.T @ M
    BMB = BM @ B;

    z = np.linalg.solve(BMB, BM @ x)

    return z