import numpy as np
import igl
import scipy as sp

'''
Computes grouping matrices for the cluster labels l, and the mesh V, T

Inputs
l - label for each tet
V - mesh vertices
T - mesh tets

Outputs
G - c x t grouping matrix
Gm - c x t grouping matrix with mass normalization
'''
def cluster_grouping_matrices(l, V, T, return_mass=False):
    """Computes the grouping matrices for a clustering of tets.

    Parameters
    ----------
    l : (t,) int numpy array
        Label for each tet
    V : (n, 3) float numpy array
        Mesh vertices
    T : (t, 4) int numpy array
        Mesh tets
    return_mass : bool
        Whether to return the mass of each cluster, tet and the mass fraction of each tet in its cluster.

    Returns
    --------
    G : (c, t) scipy sparse csc matrix
        Grouping matrix
    Gm : (c, t) scipy sparse csc matrix
        Grouping matrix with mass normalization
    mc : (c,) float numpy array
        Mass of each cluster, only returned if return_mass is True
    mt : (t,) float numpy array
        Mass of each tet, only returned if return_mass is True
    f : (t,) float numpy array
        Mass fraction of each tet in its cluster, only returned if return_mass is True

    """
    t = T.shape[0]
    c = l.max() + 1
    assert(T.shape[1] == 4)
    I= l
    J = np.arange(t)
    mt = igl.volume(V, T)
    if mt.ndim==0:
        mt = mt[None]
    mc = np.bincount(l, mt) #mass of each cluster
    Mci = sp.sparse.diags(1/mc, 0)
    Mt = sp.sparse.diags(mt,  0)

    VV = np.ones(t)
    G = sp.sparse.csc_matrix((VV, (I, J)), shape=(c, t))

    Gm = Mci @ G @ Mt

    if return_mass:
         f = mt / mc[l]
         return G, Gm, mc, mt, f
    return G, Gm

