
import scipy as sp
import numpy as np

import fast_cd_pyb as fcd

''' 
Simulation state for a Fast CD Simulation.
For Fast CD, the state is comprised of 4 quantities:
1. z_curr: the current state of the reduced secondary motion
2. p_curr: the current state of the rig parameters
3. z_prev: the previous state of the reduced secondary motion
4. p_prev: the previous state of the rig parameters
'''
class fast_cd_state(fcd.cd_sim_state):
    '''
    Sets both current and previous simulation state to
    (z0, p0)ate of simulation to z_curr and p_curr

    Inputs:
    z_curr - m x 1 reduced space coefficients for subspace sim
    p_curr - 12b x 1 rig parameters

    (Optional)
    z_prev - m x 1 reduced space coefficients for subspace sim
    p_prev - 12 b x 1 rig parameters for subspace sim
    '''
    def __init__(self, z_curr, p_curr, z_prev=None, p_prev=None):
        z_curr = z_curr.copy()
        p_curr = p_curr.copy()

        if z_prev is None:
            z_prev = z_curr.copy()
        if p_prev is None:
            p_prev = p_curr.copy()

        # call constructor from parent
        super().__init__(z_curr, z_prev, p_curr, p_prev)

        # fcd.cd_sim_state(z_curr, z_prev, p_curr, p_prev

    '''
    Updates the simulation state
    Inputs:
    z - m x 1 reduced space coefficients for subspace sim
    p - 12b x 1 rig parameters
    '''
    def update(self, z, p):
        self.z_prev = self.z_curr.copy()
        self.p_prev = self.p_curr.copy()

        self.z_curr = z.copy()
        self.p_curr = p.copy()


'''
Fast CD Simulation, implementation of https://www.dgp.toronto.edu/projects/fast_complementary_dynamics_site/
'''
class fast_cd_sim():

    '''
    Inputs:
    V - n x 3 vertex positions
    T - F x 4 tet indices
    B - 3n x m subspace matrix
    l - F x 1 cluster labels
    J - 3n x 12b LBS rig jacobian
    mu - scalar first lame parameter
    rho - scalar density
    h - scalar timestep

    max_iters - maximum number of iterations for the local-global solver
    threshold - convergence threshold for local global solver
    read_cache - whether to read simulation precomp from cache (default=False)
    cache_dir - directory to read/write cache from/to (default="")
    Aeq - c x m constraint matrix (default empty)

    '''
    def __init__(self, V, T, B, l, J, mu=1e5, rho=1, h=1e-2, max_iters=10, threshold=1e-6,
                 read_cache=False, cache_dir="", Aeq=None):
        #These parameters need to be global member variables otherwise their memory is destroyed
        self.solver_params = fcd.local_global_solver_params(False, 30, 1e-8)
        if Aeq is None:
            self.Aeq = sp.sparse.csc_matrix((0, 0))
        self.Jsp = sp.sparse.csc_matrix(J)
        self.sim_params = fcd.fast_cd_arap_sim_params(V, T, B, l, self.Jsp, self.Aeq, mu, h, 200.)

        write_cache = True
        self.sim = fcd.fast_cd_arap_sim(cache_dir, self.sim_params, self.solver_params, read_cache, write_cache)


        return

    '''
    Steps simulation state forward
    Inputs:
        p - 12b x 1 next timestep rig parameters
        state - current fast_cd_state object
    Optional:
        f_ext - m x 1 external force (default=0)
        bc - c x 1 boundary constraints (default=None). Only valid if fast_cd_sim.Aeq is non-empty
        z - m x 1 first guess for local global solver
    Returns:
        z_next - m x 1 next timestep reduced space coefficients for sim   
    '''
    def step(self,  p, state, z=None, f_ext=None, bc=None):
        if f_ext is None:
            f_ext = np.zeros((state.z_curr.shape[0], 1))
        if bc is None:
            bc = np.array([[]], dtype=np.float64).T
        else:
            assert(bc.shape[0] == self.Aeq.shape[0])
        if z is None:
            z = state.z_curr

        # st = fcd.cd_sim_state(state.z_curr, state.z_prev,
        #                       state.p_curr, state.p_prev)
        # state2 = fcd.cd_sim_state(state)
        z = self.sim.step(z, p, state, f_ext, bc)[:, None]

        return z

    # def step(self, z, p, z_curr, p_curr, z_prev, p_prev, f_ext=None, bc=None):
    #     st = fcd.cd_sim_state(z_curr, z_prev, p_curr, p_prev)
    #     z = self.step_state(z, p, st, f_ext, bc)
    #     return z
    #

