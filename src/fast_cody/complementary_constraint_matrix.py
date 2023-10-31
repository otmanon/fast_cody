import igl
import scipy as sp

import fast_cody as fc


def complementary_constraint_matrix(V, T, J, dt=None):
    """ Computes the complementarity constraint matrix
        ```
        C = J D M
        ```
        Where J is the rig jacobian, D is the momentum leaking matrix and M is the mass matrix.
        The momentum leaking matrix is computed via surface diffusion using the timestep dt, which is set to 1/l^2 by default.

    Parameters
    ----------
    V : (n, 3) float numpy array
        Mesh vertices
    T : (t, 4) int numpy array
        Mesh tets
    J : (3n, 12m) float numpy array
        Rig jacobian matrix
    dt : float
        Timestep used for momentum leaking matrix, (default=1/l^2)

    Returns
    --------
    C : (12m, 3n) float numpy array
        Complementarity constraint matrix

    """
    M = igl.massmatrix(V, T)
    Me = sp.sparse.kron(sp.sparse.identity(3), M)
    D = fc.momentum_leaking_matrix(V, T, dt=dt)

    C =  (Me @ D @ J).T


    return C