import igl
import numpy as np
import scipy as sp

from .diffuse_weights import diffuse_weights


def momentum_leaking_matrix(V, T, dt=None, pow=1):
    """
    Constructs the momentum leaking matrix, that fudges the CD constraint to allow momentum to leak from
    the rig to the mesh. This is a diagonal matrix with entries ranging from 0 (full momentum leak), to 1 (no momentum leak).
    This leaking matrix is computed via a surface diffusion, where values at the surface have momentum leaking value set to 0,
    and smoothly increases to 1 at the interior.


    Parameters
    ----------
    V : (n, 3) float numpy array
        Vertex positions
    T : (t, 4) int numpy array
        Tet indices
    dt : float
        Used in diffusion (default 1/l^2 where l is the mean edge lengths)
    pow : float
        Power to raise the diffusion weights to (default 1)

    Returns
    -------
    D : (n, n) scipy sparse matrix
        Diagonal sparse matrix with entries varying from 0 (momentum-fully leaking) to 1 (momentum not leaking) for each vertex.
    """
    F = igl.boundary_facets(T)
    M = igl.massmatrix(V, T)
    Me = sp.sparse.kron(sp.sparse.identity(3), M)
    bI = np.unique(F)
    phi = np.ones((bI.shape[0], 1))
    d = 1 - np.power(diffuse_weights(V, T, phi, bI, dt=dt), pow)

    # import polyscope as ps
    # ps.init()
    # m = ps.register_volume_mesh("mesh", V, T)
    # m.add_scalar_quantity("d", d[:, 0], enabled=True)
    # ps.show()
    D = sp.sparse.kron(sp.sparse.identity(3), sp.sparse.diags(d[:, 0]))


    return D
