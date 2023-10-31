import numpy as np
import scipy as sp
import os
import igl

from .laplacian_eigenmodes import laplacian_eigenmodes
from .skinning_clusters import skinning_clusters
from .lbs_jacobian import lbs_jacobian
from .orthonormalize import orthonormalize


def skinning_subspace(X, T, num_modes, num_clusters,
                      cache_dir=None, read_cache=False,
                      ortho=True, mu=None, C=None, constraint_enforcement="optimal"):
    """
    Constructs a physics subspace corresponding with skinning eigenmodes and skinning clusters

    Parameters
    ----------
    X : (n, 3) float numpy array
        Vertex positions
    T : (t, 4) int numpy array
        Tet indices
    num_modes : int
        Number of modes to use
    num_clusters : int
        Number of clusters to use
    cache_dir : str
        Directory to cache results in
    read_cache : bool
        Whether to read from cache or not
    ortho : bool
        Whether to orthonormalize the subspace
    mu : float numpy array
        Per-tet conducivity. If None, sets it to 1.0 everyewhere
    C : (3n, c) float numpy array
        Constraint matrix we desire on our weights s.t. C.T @ W = 0
    constraint_enforcement : str
        Method of enforcing constraint. Either "project" or "optimal"

    Returns
    -------
    B : (3n, m) float numpy array
        Subspace matrix/Eigenvectors of laplacian.
    l : (t, 1) int numpy array
        Cluster indices
    W : (n, m) float numpy array
        Skinning weights


    """

    dim = X.shape[1]

    if cache_dir is not None:
        cache_dir = os.path.join(cache_dir, "./")
        os.makedirs(cache_dir, exist_ok=True)
    if read_cache and cache_dir is not None:
        assert (os.path.exists(cache_dir) and "cache directory " + cache_dir + " we are trying to read from does not exist")
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
        [W, E] = laplacian_eigenmodes(X, T, num_modes, read_cache=False, mu=mu, J=C, constraint_enforcement=constraint_enforcement)

        B = lbs_jacobian(X, W)

        # WeightsViewer(X, T, B)
        M = sp.sparse.kron(sp.sparse.identity(3), igl.massmatrix(X, T))

        # if ortho:
        #     B = orthonormalize(B, M)
        l = skinning_clusters(W, E, T, num_clusters, l=2, num_clustering_features=num_modes)

        if (cache_dir is not None):
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_dir + "l.npy", l)
            np.save(cache_dir + "B.npy", B)
            np.save(cache_dir + "W.npy", W)

    return B, l, W