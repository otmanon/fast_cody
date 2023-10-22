

import numpy as np
import igl

import fast_cd
from .laplacian import laplacian

'''
Diffuses phi quantities on tet mesh Vv, Tv at nodes bI fro time 

Inputs
------
Vv: nx3 volume vertices
Tv: tx4 volume tetrahedra
phi: c x b quantity to diffuse
bI: c x b indices at diffusion points

returns
W: n x b  diffused quantities over entire mesh
'''
def diffuse_weights(Vv, Tv, phi, bI,  dt=None ):

    if (dt is None):
        dt = np.mean(igl.edge_lengths(Vv, Tv)) ** 2

    L = laplacian(Vv, Tv)
    M = igl.massmatrix(Vv, Tv)

    Q = L * dt + M

    ii = np.setdiff1d(  np.arange(Q.shape[0]), bI)
    # selection matrix for indices bI
    Qii = Q[ii, :][:, ii]
    Qib = Q[ii, :][:, bI]

    Wii = fast_cd.umfpack_lu_solve(Qii, -Qib @ phi)
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
    W = (W - np.min(W, axis=0)[:, None]) / (np.max(W, axis=0)[:, None] - np.min(W, axis=0)[:, None])
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