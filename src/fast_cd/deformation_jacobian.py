import igl
import scipy as sp
import numpy as np


def deformation_jacobian(V, T):
    G = igl.grad(V, T)
    t=T.shape[0]
    I = sp.sparse.identity(V.shape[1])

    Ge = sp.sparse.block_diag((G, G, G))


    imat = np.tile(np.arange(0, 9), (t, 1)) + np.arange(0, t)[:, np.newaxis] * 9;

    jmat = np.tile(np.array([0, 3 * t, 6 * t, 1 * t, 4 * t, 7 * t, 2 * t, 5 * t, 8 * t]), (t, 1)) + np.arange(0, t)[:, np.newaxis]

    vals = np.ones(imat.shape)
    P = sp.sparse.coo_matrix((vals.flatten(), (imat.flatten(), jmat.flatten())), shape=(9 * t, 9 * t)).tocsc()
    J = P @ Ge

    # f = J@V.flatten(order="F")
    # F = f.reshape((t, 9))

    return J

