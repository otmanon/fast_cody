import igl
import scipy as sp
import numpy as np

def curve_laplacian(V, E):

    l = igl.edge_lengths(V, E)

    # each edge has four contributions to the Laplacian

    I = np.hstack((E[:, 0], E[:, 1], E[:, 0], E[:, 1]))
    J  = np.hstack((E[:, 1], E[:, 0], E[:, 0], E[:, 1]))

    VV = np.hstack((-1.0/l, -1.0/l, 1.0/l, 1.0/l))

    L = sp.sparse.csc_matrix((VV, (I, J)), shape=(V.shape[0], V.shape[0]))

    return L


from .deformation_jacobian import deformation_jacobian
def tet_laplacian(X, T, mu=None):

    if mu is None:
        mu = np.ones(T.shape[0])
    else:
        assert(type(mu) == np.ndarray)
        assert(mu.shape[0] == T.shape[0])

    muv = np.kron(mu, np.ones(9))
    Muv = sp.sparse.diags(muv)

    J = deformation_jacobian(X, T)
    a = igl.volume(X, T)
    A = sp.sparse.kron(sp.sparse.diags(a), sp.sparse.identity(9))

    L =  J.T @ A @ Muv @ J
    L = L[:X.shape[0], :][:, :X.shape[0]]
    return L

def laplacian(X, T, mu=None):
    if T.shape[1] == 2:
        if mu is not None:
            print("Warning: mu is ignored for curve laplacian")
        return curve_laplacian(X, T)
    if T.shape[1] == 3:
        if mu is not None:
            print("Warning: mu is ignored for cotan laplacian")
        return igl.cotmatrix(X, T)
    elif T.shape[1] == 4:
        return tet_laplacian(X, T, mu=mu)
