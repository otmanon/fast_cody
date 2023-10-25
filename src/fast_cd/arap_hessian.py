import scipy as sp
import numpy as np
import igl


from .deformation_jacobian import deformation_jacobian


def arap_hessian(V, F, mu=None, U=None):

    """Computes ARAP Hessian

    Parameters
    ----------
    V : (n, 3) numpy float array
        Rest vertex geometry
    F : (f, 4) numpy int array
        Tetrahedron indices
    mu : float or (f, 1) numpy float array or None
        First lame parameter (e.g. stiffness). if None, then sets it to 1 for all tets.
    U : (n, 3) numpy float array or None
        Deformed geometry where to evaluate the hessian. If None, then U=V.

    Returns
    -------
    H : (n*3, n*3) scipy sparse csc matrix
        ARAP Hessian at U
    """


    def dpsidF2(F):
        # F  is organized in a vecotrized way, #tets x d x d
        if (len(F.shape) == 2):
            F = F[None, :, :]
        d = F.shape[1]
        n = F.shape[0]
        [U, S, V] = np.linalg.svd(F)
        # V = Vt.transpose([0, 2, 1])
        T0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # (1/ np.sqrt(2)) * U * [] * V
        T0 = (1 / np.sqrt(2)) * U @ T0 @ V

        T1 = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]])
        T1 = (1 / np.sqrt(2)) * U @ T1 @ V

        T2 = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]])
        T2 = (1 / np.sqrt(2)) * U @ T2 @ V

        t0 = np.reshape(T0, (n, d * d, 1))
        t1 = np.reshape(T1, (n, d * d, 1))
        t2 = np.reshape(T2, (n, d * d, 1))

        s0 = np.reshape(S[:, 0], (n, 1, 1))
        s1 = np.reshape(S[:, 1], (n, 1, 1))
        s2 = np.reshape(S[:, 2], (n, 1, 1))

        H = 2 * np.tile(np.identity(9), (n, 1, 1))

        H -= (4 / (s0 + s1)) * (t0 @ t0.transpose(0, 2, 1))
        H -= (4 / (s1 + s2)) * (t1 @ t1.transpose(0, 2, 1))
        H -= (4 / (s0 + s2)) * (t2 @ t2.transpose(0, 2, 1))

        return H


    dim = V.shape[1]
    # B = deformation_jacobian(V, F);
    if (mu is None):
        mu = np.ones((F.shape[0]))
    elif (np.isscalar(mu)):
        mu = mu* np.ones((F.shape[0]))
    else:
        assert(mu.shape[0] == F.shape[0])

    if U is None:
        U = V.copy() # assume at rest

    muv = np.kron(mu, np.ones((dim*dim)))
    Mu = sp.sparse.diags(muv)
    B = deformation_jacobian(V, F)
    vecF = B @ U.flatten(order="F")

    defograd = np.reshape(vecF, (F.shape[0], dim, dim))
    dpdF2 = dpsidF2(defograd)

    dpdF2 = sp.sparse.block_diag(dpdF2)

    vol = igl.volume(V, F)
    A = sp.sparse.kron(sp.sparse.diags(vol), sp.sparse.identity(9))
    H = B.transpose() @ A @ Mu @ dpdF2 @ B
    return H

