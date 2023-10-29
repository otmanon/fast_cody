import igl
import scipy as sp
import numpy as np



def laplacian(X, T, mu=None):
    """
    Computes the Laplacian of a d-simplex. With d being 2, 3 or 4.

    Parameters
    ----------
    V : (n, 3) numpy float array
        Vertex positions
    T : (t, d)  (d=2, 3, 4) numpy int array
        Simplex indices
    mu : (t,) numpy float array
        Per-simplex conductivity

    Returns
    -------
    L : (n, n) scipy sparse float array
        Laplacian matrix
    """
    if mu is None:
        mu = np.ones(T.shape[0])
    elif np.isscalar(mu):
        mu = mu * np.ones(T.shape[0])
    else:
        assert (type(mu) == np.ndarray)
        assert (mu.shape[0] == T.shape[0])

    if T.shape[1] == 2:
        l = igl.edge_lengths(X, T)
        I = np.hstack((T[:, 0], T[:, 1], T[:, 0], T[:, 1]))
        J = np.hstack((T[:, 1], T[:, 0], T[:, 0], T[:, 1]))
        VV = np.hstack((-1.0 / l * mu, -1.0 / l * mu, 1.0 / l * mu, 1.0 / l * mu))
        L = sp.sparse.csc_matrix((VV, (I, J)), shape=(X.shape[0], X.shape[0]))
        return L
    if T.shape[1] == 3 or T.shape[1] == 4:
        muv = np.kron(mu, np.ones(3))
        Muv = sp.sparse.diags(muv)
        J = igl.grad(X, T)
        a = igl.volume(X, T)
        A = sp.sparse.kron(sp.sparse.identity(3), sp.sparse.diags(a))
        L = J.T @ A @ Muv @ J
        L = L[:X.shape[0], :][:, :X.shape[0]]
        return L
    else:
        raise NotImplementedError("Laplacian not implemented for dimension %d" % T.shape[1])




