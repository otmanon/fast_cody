import scipy as sp

from scipy.sparse import hstack


def closest_orthogonal_subspace(A, B, M=None):
    """ Projects A to its closest linear space that satisfies the constraints AB = 0
        Solves the optimization problem :
        ```
        C = argmin_C ||C - A||_M^2 s.t. CB = 0
        ```
    Parameters
    ----------
    A : (n, d) float numpy array
        Linear space
    B : (c, d) float numpy array
        Linear space with which we want to be orthogonal
    M : (n, n) float sparse matrix
        Mass metric with which we measure orthogonality (default=identity(n))

    Returns
    -------
    C : (n, d) float numpy array
        Projected linear space that is close to A while satisfying CB=0

    """
    if M is None:
        M = sp.sparse.identity(A.shape[0])
    c = B.shape[0]

    BMA = B.T @ M @ A


    BMB = B.T @ M @ B
    BMBBMA = sp.sparse.linalg.spsolve(BMB, BMA)
    A2 = A - B @ BMBBMA
    return A2

