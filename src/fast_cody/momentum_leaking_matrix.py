import igl
import numpy as np
import scipy as sp

from .diffuse_weights import diffuse_weights

'''
Constructs the momentum leaking matrix, that fudges the CD constraint
in order to let momentum leak from the rig to the mesh

Inputs:
V - n x 3 mesh geometry
T - t x 4 tet indices

Optional:
dt - float,  used in diffusion (default 1/l^2 where l is the mean edge lengths)
expand_dim - bool, if True returns a 3nx3n diagonal matrix, 
Outputs:
D - n x n diagonal sparse matrix with entries varying from 0 (momentum-fully leaking) to 1 (momentum not leaking) for each vertex
'''
def momentum_leaking_matrix(V, T, dt=None, pow=1, expand_dim=True):

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
