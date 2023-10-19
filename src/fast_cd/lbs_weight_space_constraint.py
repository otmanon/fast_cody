import scipy as sp
import numpy as np
from .orthonormalize import orthonormalize
'''
Linear Blend Skinning Weight Space Constraint

converts a constraint matrix that acts on displacements 3n,
to act instead on the n skinning weights 

Inputs:
V: n x 3 vertices
W: n x d skinning weights... dont think we need this
C: c x 3n constraints

Outputs:
C2 : n x c' constraints in weight space with redundnat rows removed

'''
def lbs_weight_space_constraint(V, C, M=None):
    C = C.T
    n = V.shape[0]
    d = V.shape[1]

    v = np.ones((n, 1))

    A = np.zeros((0, n))
    for i in range(0, d):
        Id = np.arange(0, n) + i * n
        Jd = np.arange(0, n)
        Pd = sp.sparse.coo_matrix((v.flatten(), (Id, Jd)), shape=(3*n,n))

        for j in range(0, d):
            Vj = V[:, j]
            Adj = C.T @ Pd @ sp.sparse.diags(Vj, 0)
            A = np.vstack([A, Adj])
        #j+1
        Ad1 = C.T @ Pd
        A = np.vstack([A, Ad1])

    # form basis, remove redundant rows
    W = A

    W2 = orthonormalize(W.T).T
    # [u, s, v] = np.linalg.svd(W)
    # ii = np.where(s > 1e-9)[0]
    # W2 = u[:, ii]# @ np.diag(s[ii])




    return W2