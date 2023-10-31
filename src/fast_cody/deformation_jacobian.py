import igl
import scipy as sp
import numpy as np


def deformation_jacobian(V, T):
    """ Computes the deformation Jacobian of a tetrahedral mesh.
    The resulting jacobian J is used to obtain the deformation gradient from the positions.

    Parameters
    ----------
    V : (n, 3) float numpy array
        Mesh vertices
    T : (t, 4) int numpy array
        Mesh tets

    Returns
    --------
    J : (9t, 3n) scipy sparse csc matrix
        Deformation Jacobian matrix


    Examples
    --------
    Obtain a stacked t x 3 x 3 list of deformation gradients from each tet, from n x 3 positions U
    ```
    >>>import fast_cody as fcd
    >>> [V, F, T] = fcd.fcd.get_data('cd_fish.msh')
    >>>J = fcd.deformation_jacobian(X, T)
    >>>f = J @ U.flatten(order='F')
    >>>F = f.reshape(-1, 3, 3)
    ```



    """
    G = igl.grad(V, T)
    t=T.shape[0]
    I = sp.sparse.identity(V.shape[1])

    Ge = sp.sparse.block_diag((G, G, G))


    imat = np.tile(np.arange(0, 9), (t, 1)) + np.arange(0, t)[:, np.newaxis] * 9;

    jmat = np.tile(np.array([0, 3 * t, 6 * t, 1 * t, 4 * t, 7 * t, 2 * t, 5 * t, 8 * t]), (t, 1)) + np.arange(0, t)[:, np.newaxis]

    vals = np.ones(imat.shape)
    P = sp.sparse.coo_matrix((vals.flatten(), (imat.flatten(), jmat.flatten())), shape=(9 * t, 9 * t)).tocsc()
    J = P @ Ge



    return J

