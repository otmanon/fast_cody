
import igl
import scipy as sp
from scipy.sparse import hstack, vstack


import os
import time
import numpy as np

from .laplacian import laplacian
from .project_out_subspace import project_out_subspace
from .orthonormalize import orthonormalize
from .eigs import eigs

def laplacian_eigenmodes(V, T, m, read_cache=False, cache_dir=None, J=None,
                         mu=None, constraint_enforcement="optimal"):
    """ Computes Laplacian Eigenmodes for a given mesh.

    Parameters
    ----------
    V : (n, 3) float numpy array
        Vertex positions
    T : (F, 4) int numpy array
        Tet indices
    m : int
        Number of modes to compute
    read_cache : bool
        Whether to read from cache or not (default False)
    cache_dir : str
        Directory to cache results in (default None)
    J : (c x n) float numpy array
        Constraint matrix we desire on our weights s.t. J @ W = 0 (default None)
    mu : float
        Per-tet conducivity. If None, sets it to 1.0 everyewhere (default None)
    constraint_enforcement : str
        Method of enforcing constraint. Either "project" or "optimal" (default "optimal")

    Returns
    -------
    B : (n, m) float numpy array
        Subspace matrix/Eigenvectors of laplacian.
    E : (m, 1) float numpy array
        Eigenvalues of each eigenvector
    """
    if read_cache:
        B = np.load(cache_dir + "/B.npy")
        E = np.load(cache_dir + "/E.npy")
        return B, E
    else:
        L = laplacian(V, T, mu=mu)
        M = igl.massmatrix(V, T)
        L =  L + 1e-8 * M
        if constraint_enforcement == "optimal":
            if J is not None:
                c = J.shape[0]
                Z = sp.sparse.csc_matrix((c, c))
                L = vstack((hstack((L, J.T)), hstack((J, Z )))).tocsc()
                M = sp.sparse.block_diag((M, Z)).tocsc()
        print("Computing eigenmodes... may take a while...")
        start = time.time()
        [E, B] = eigs(L, M=M, k=m)
        print("Done computing eigenmodes! Took, ", time.time() - start, " seconds")

        n = V.shape[0]
        if J is not None:
            B = B[:n, :]

        B = np.real(B)
        E = np.real(E)

        if constraint_enforcement == "project":
            B = project_out_subspace(B, J.T)
            E = np.diag(B.T @ L @ B)
            # WeightsViewer(V, T, B)
            print("Done projecting out constraints from eigenmodes")

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_dir + "/B.npy", B)
            np.save(cache_dir + "/E.npy",E)

        B = orthonormalize(B)

    return B, E