
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
''' 
Constructs a physics subspace corresponding with skinning eigenmodes and skinning clusters
Inputs:
    X: V x 3 vertex positions
    T: F x 4 tet indices
    num_modes: number of modes to use

(Optional)
    cache_dir: directory to cache results in (default None)
    read_cache: whether to read from cache or not (default False)
    C :  c x 3n constarint matrix that acts on the skinning weights (default None)
    constraint_enforcement : method of enforcing constraint. Either "project" or "optimal"
                        for python, default is "project", because optimal makes eigendecomposition take way too long.
Outputs:
    B: 3n x 12m  subspace matrix
    W: n x m skinning weights
'''
def laplacian_eigenmodes(V, T, num_modes, read_cache=False, cache_dir=None, J=None,
                         mu=None, constraint_enforcement="project"):
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
        [E, B] = eigs(L, M=M, k=num_modes) #sp.sparse.linalg.eigs(L, M=M, k=num_modes, sigma=0, which='LM')
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