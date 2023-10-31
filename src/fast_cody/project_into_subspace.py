import scipy as sp
import numpy as np
'''
projects x into a subspace B via least squares
'''
def project_into_subspace(x, B, M=None):
    """ Projects x into a subspace B via least squares.
    Solves the following optimization problem:
        ```
            argmin_z ||Bz - x||_2^2
        ```
    Parameters
    ----------
    x : (n, 1) float numpy array
        Vector to be projected into the subspace
    B : (n, m) float numpy array
        Subspace to project x into
    M : (m, m) float numpy array
        Mass matrix defining the metric for projection. If None, set to identity matrix

    Returns
    -------
    z : (m, 1) float numpy array
        Projection of x into the subspace B
    """
    if M is None:
        M = sp.sparse.identity(x.shape[0])
    if M.shape[0] == B.shape[0]//3:
        M = sp.sparse.kron(sp.sparse.identity(3), M)
    BM = B.T @ M
    BMB = BM @ B;

    z = np.linalg.solve(BMB, BM @ x)
    return z