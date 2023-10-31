

import numpy as np
import igl

import fast_cody
from .laplacian import laplacian


def diffuse_weights(Vv, Tv, phi, bI,  dt=None, normalize=True):
    """ Performs a diffusion on the tet mesh Vv, Tv at nodes bI for time dt.

    Parameters
    ----------
    Vv : (n, 3) float numpy array
        Mesh vertices
    Tv : (t, 4) int numpy array
        Mesh tets
    phi : (c, b) float numpy array
        Quantity to diffuse
    bI : (c, b) int numpy array
        Indices at diffusion points
    dt : float
        Time to diffuse for
    normalize : bool
        Whether to normalize the weights

    Returns
    -------
    W : (n, b) float numpy array
        Diffused quantities over entire mesh

    """

    if (dt is None):
        dt = np.mean(igl.edge_lengths(Vv, Tv)) ** 2

    L = laplacian(Vv, Tv)
    M = igl.massmatrix(Vv, Tv)

    Q = L * dt + M

    ii = np.setdiff1d(  np.arange(Q.shape[0]), bI)
    # selection matrix for indices bI
    Qii = Q[ii, :][:, ii]
    Qib = Q[ii, :][:, bI]

    Wii = fast_cody.umfpack_lu_solve(Qii, -Qib @ phi)
    W = np.zeros((L.shape[0], Wii.shape[1]))
    W[ii, :] = Wii
    W[bI, :] = phi
    # W = gpt.min_quad_with_fixed(L*dt + M, k=bI, y=phi)

    # normalize weights so that max is 1 and min is 0
    # W = (W - np.min(W, axis=0)[:, None]) / (np.max(W, axis=0)[:, None] - np.min(W, axis=0)[:, None])

    # ps.init()
    # # pc = ps.register_point_cloud("pc", Vs)
    # # pc_CI = ps.register_point_cloud("pc_cI", Vs[CI])
    # mesh = ps.register_volume_mesh("mesh", Vv, Tv)
    # mesh.add_scalar_quantity("w", W.flatten())
    # ps.show()


    # normalize between 0 and 1
    if W.ndim == 1:
        W = W[:, None]
    if normalize:
        W = (W - np.min(W, axis=0)) / (np.max(W, axis=0) - np.min(W, axis=0))
    # WeightsViewer(Vs, Ts, Ws, period=1)
    # WeightsViewer(Vv, Fv, W)

    # ps.init()
    # volms = ps.register_volume_mesh("vol", Vv, Tv)
    # volms.add_scalar_quantity("W", W[:, 0])
    #
    # ps.show()

    # ps.init()
    # mesh = ps.register_volume_mesh("mesh", Vv, Tv)
    # mesh.add_scalar_quantity("weights", W[:, 0], enabled=True, cmap='coolwarm')
    # ps.show()
    # WeightsViewer(Vv, Tv, W)
    return W