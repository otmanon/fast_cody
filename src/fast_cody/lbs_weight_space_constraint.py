import scipy as sp
import numpy as np
from .orthonormalize import orthonormalize
def lbs_weight_space_constraint(V, C):
    """ Rewrites a linear equality constraint that acts on per-vertex displacements (CU(W) = 0)
        to instead act on the per-vertex skinning weights  (AW = 0).

    Parameters
    ----------
    V : (n, d) float numpy array
        Mesh vertices
    C : (c, dn) float numpy array
        Linear equality constraint matrix that acts on per-vertex displacements

    Returns
    -------
    A : (n, c') float numpy array
        Linear equality constraint matrix that acts on per-vertex skinning weights
    """
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
        Ad1 = C.T @ Pd
        A = np.vstack([A, Ad1])

    W = A
    W2 = orthonormalize(W.T).T



    return W2