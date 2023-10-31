import numpy as np
import scipy as sp

from scipy.sparse import vstack, hstack


from fast_cody.umfpack_lu_solve import umfpack_lu_solve
'''
Performs a least squares projection on subspace A so that it does not span space B

Finds A2 as close as possible to A such that A2 is orthogonal to B

||C - A||^2_F st. C^T B = 0
C'MC - 2 C'MA + mu' C'B

[ M  B ] [ C ]  = [ MA ]
[ B' 0 ] [ mu ]   [ 0 ]

C = M^-1 (MA - B mu)
B' C = 0 -> B' M^-1 (MA - B mu) = 0 ->  mu = ( B' M^-1 B) B' A 
C = A - M^-1 B  (B' M^-1 B)^(-1) B' A 


A - n x m subspace matrix
B - n x k subspace matrix, where k < m

Returns:
    A2 - n x m' subspace matrix, where m' <= m
'''
def project_out_subspace(A, B, M=None):
    """ Performs a least squares projection on subspace A so that it does not span space B.
    Finds C as close as possible to A such that C is orthogonal to B.
    ```
        argmin_C ||C - A||^2_F st. C^T B = 0
        ||C - A||^2_F st. C^T B = 0
        C'MC - 2 C'MA + mu' C'B

        [ M  B ] [ C ]  = [ MA ]
        [ B' 0 ] [ mu ]   [ 0 ]

        C = M^-1 (MA - B mu)
        B' C = 0 -> B' M^-1 (MA - B mu) = 0 ->  mu = ( B' M^-1 B) B' A
        C = A - M^-1 B  (B' M^-1 B)^(-1) B' A
    ```

    Parameters
    ----------
    A : (n, m) float numpy array
        Subspace of interes.
    B : (n, k) float numpy array
        Subspace to project out. We want C to be orthogonal to B
    M : (n, n) float numpy array
        Mass matrix defining the metric for projection. If None, set to identity matrix

    Returns
    -------
    C : (n, m) float numpy array
        Projection of A

    """


    assert(B.shape[0] == A.shape[0])
    if M is None:
        M = sp.sparse.identity(A.shape[0]).tocsc()

    Z = sp.sparse.csc_matrix((B.shape[1], B.shape[1]))
    Bsp = sp.sparse.csc_matrix(B)
    Q = vstack((hstack([M, Bsp]), hstack((Bsp.T, Z))))
    # Q = vstack(hstack((M, B)), hstack((B.T, Z)))

    z = sp.sparse.csc_matrix((B.shape[1], A.shape[1]))

    Asp = sp.sparse.csc_matrix(A)
    rhs = vstack(( M @ Asp, z))

    Cmu = umfpack_lu_solve(Q, rhs.todense())
    #sp.sparse.linalg.spsolve(Q, rhs)
    C = Cmu[:A.shape[0], :]
    # Cd = C.toarray()
    return C