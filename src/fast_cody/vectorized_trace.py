import numpy as np
import scipy as sp

'''
Given a flattened stack of n  dxd matrices, computes the trace operator
that, when applied to this stack, returns another flattened stack of n values containing the traces
of each of these matrices.
Assumes column-major flattening.
'''
def vectorized_trace(n, d):
    """
    Given a flattened stack of n  dxd matrices, computes the trace operator
    that, when applied to this stack, returns another flattened stack of n values containing the traces
    of each of these matrices.

    Parameters
    ----------
    n : int
        Number of matrices
    d : int
        Dimension of matrices

    Returns
    -------
    T : (n, n*d*d) scipy sparse matrix

    """
    # trace of matrix i will have elements i, then n+i + 1, 2n+i+2

    ii = np.arange(n * d * d)
    Mi = np.reshape(ii, (n * d, d))

    Mii = np.reshape(Mi, (n, d, d))
    Mii = np.reshape(Mii, (d * n, d))

    mask = np.identity(d, dtype=bool)
    maskii = np.tile(mask, (n, 1))
    j = Mii[maskii]
    i = np.repeat(np.arange(n), d)

    v = np.ones((i.shape[0]))
    T = sp.sparse.csc_matrix((v, (i, j)), (n, n * d * d))


    return T