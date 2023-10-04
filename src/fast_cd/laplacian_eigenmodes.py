
import igl
import scipy as sp
from scipy.sparse import hstack, vstack


import os
import time
import numpy as np

from .laplacian import laplacian


def laplacian_eigenmodes(V, T, num_modes, read_cache=False, cache_dir=None, J=None, mu=None):
    if read_cache:
        B = np.load(cache_dir + "/B.npy")
        E = np.load(cache_dir + "/E.npy")
        return B, E
    else:
        L = laplacian(V, T, mu=mu)
        M = igl.massmatrix(V, T)
        L =  L + 1e-8 * M
        if J is not None:
            c = J.shape[0]
            Z = sp.sparse.csc_matrix((c, c))
            L = vstack((hstack((L, J.T)), hstack((J, Z )))).tocsc()
            M = sp.sparse.block_diag((M, Z)).tocsc()

        print("Computing eigenmodes... may take a while...")
        start = time.time()
        [E, B] = sp.sparse.linalg.eigs(L, M=M, k=num_modes, sigma=0, which='LM')
        print("Done computing eigenmodes! Took, ", time.time() - start, " seconds")
        n = V.shape[0]
        if J is not None:
            B = B[:n, :]

        B = np.real(B)
        E = np.real(E)

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_dir + "/B.npy", B)
            np.save(cache_dir + "/E.npy",E)

    return B, E