import igl
import scipy as sp

import fast_cd as fc

'''
Inputs:
V - n x 3 mesh geometry
T - t x 4 tet indices
J - 3n x 12m rig jacobian matrix

Optional:
dt - timestep used for momentum leaking matrix, (default=1/l^2)

Outputs:
C - 12m x 3n complementary dynamics constraint matrix
  
'''
def complementary_constraint_matrix(V, T, J, dt=None):
    M = igl.massmatrix(V, T)
    Me = sp.sparse.kron(sp.sparse.identity(3), M)
    D = fc.momentum_leaking_matrix(V, T, dt=dt)

    C =  Me @ D @ J

    return C.T