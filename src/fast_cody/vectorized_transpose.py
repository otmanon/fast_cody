import numpy as np
import scipy as sp
'''
Given a flattened stack of n  dxd matrices, computes the transpose operator
that, when applied to this stack, returns another flattened stack of n dxd transposed versions
of the original matrices.

Assumes column-major flattening.
'''

'''
1x 1y
1z 1w

2x 2y
2z 2w

3x 3y
3z 3w
= 

'''
def vectorized_transpose(n, d):
    """
    Given a flattened stack of n  dxd matrices, computes the transpose operator

    Parameters
    ----------
    n : int
        Number of matrices
    d : int
        Dimension of matrices

    Returns
    -------
    T : (n*d*d, n*d*d) scipy sparse matrix
        Vectorized transpose operator
    """
    ii = np.arange(n*d*d)
    Mi = np.reshape(ii, (n*d, d))

    Mii = np.reshape(Mi, (n, d, d))
    Mii = Mii.transpose((0, 2, 1))
    Mii = np.reshape(Mii, (d*n, d))
    Mj = Mii.flatten()
    # transpose each block

    i = Mj;
    j = np.arange(n*d*d)
    v = np.ones(n*d*d)
    T = sp.sparse.csc_matrix((v, (i, j)), (n*d*d, n*d*d))
    return T