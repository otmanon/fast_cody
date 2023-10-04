import numpy as np
import scipy as sp
import os
import igl

from .arap_hessian import arap_hessian
from .laplacian_eigenmodes import laplacian_eigenmodes
from .skinning_clusters import skinning_clusters
from .lbs_jacobian import lbs_jacobian
from .orthonormalize import orthonormalize

''' 
Constructs a physics subspace corresponding with skinning eigenmodes and skinning clusters
Inputs:
    X: V x 3 vertex positions
    T: F x 4 tet indices
    num_modes: number of modes to use
    num_clusters: number of clusters to use
    cache_dir: directory to cache results in
    read_cache: whether to read from cache or not
Outputs:
    B: 3n x 12m  subspace matrix
    l: F x 1 cluster indices
    W: n x m skinning weights
'''
def skinning_subspace(X, T, num_modes, num_clusters,
                      cache_dir=None, read_cache=False,
                      ortho=True, mu=None, C=None):
    dim = X.shape[1]

    if cache_dir is not None:
        cache_dir = os.path.join(cache_dir, "./physical_subspace/")
        os.makedirs(cache_dir, exist_ok=True)
    if read_cache and cache_dir is not None:
        assert (os.path.exists(cache_dir), "cache directory " + cache_dir + " We are trying to read from does not exist")
        B = np.load(cache_dir + "/B.npy")


        W = np.load(cache_dir + "/W.npy")
        W = W[:, :num_modes]
        #
        i = np.arange(B.shape[1])
        ii = i.reshape((B.shape[1]//3, 3), order='F').reshape((B.shape[1]//12, 4, 3) )
        i = ii[:num_modes, :, :].reshape((num_modes*4, 3)).reshape((num_modes*12), order='F')
        B = B[:, i]
        l = np.load(cache_dir + "/l.npy")
    else:
        [W, E] = laplacian_eigenmodes(X, T, num_modes, read_cache=False, mu=mu, J=C)

        B = lbs_jacobian(X, W)
        M = sp.sparse.kron(sp.sparse.identity(3), igl.massmatrix(X, T))

        if ortho:
            B = orthonormalize(B, M)
        l = skinning_clusters(W, E, T, num_clusters, l=2)

        if (cache_dir is not None):
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_dir + "l.npy", l)
            np.save(cache_dir + "B.npy", B)
            np.save(cache_dir + "W.npy", W)
    return B, l, W