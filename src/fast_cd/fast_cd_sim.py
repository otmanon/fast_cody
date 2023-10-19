
import scipy as sp
import numpy as np

import fast_cd_pyb as fcd

class fast_cd_sim():


    def __init__(self, V, T, B, l, J, mu=1e5, rho=1, h=1e-2, max_iters=10, threshold=1e-6,
                 read_cache=False, cache_dir="", z0=None, p0=None):
        solver_params = fcd.local_global_solver_params(True, max_iters, threshold)
        Aeq = sp.sparse.csc_matrix((0, 0))
        Jsp = sp.sparse.csc_matrix(J)
        sim_params = fcd.fast_cd_arap_sim_params(V, T, B, l, Jsp, Aeq, mu, h, rho)
        write_cache = True
        self.sim = fcd.fast_cd_arap_sim(cache_dir, sim_params, solver_params, read_cache, write_cache)


        return

    def step(self, z, p, state, f_ext, bc):

        z = self.sim.step(z, p, state, f_ext, bc)
        return z

