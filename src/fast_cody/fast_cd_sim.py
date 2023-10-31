
import scipy as sp
import numpy as np

import fast_cd_pyb as fcd


class fast_cd_state(fcd.cd_sim_state):
    """
    Simulation state for a Fast Complementary Dynamics Simulation

    For Fast CD, the state is comprised of 4 quantities:
        z_curr: the current state of the reduced secondary motion
        p_curr: the current state of the rig parameters
        z_prev: the previous state of the reduced secondary motion
        p_prev: the previous state of the rig parameters
    """
    def __init__(self, z_curr, p_curr, z_prev=None, p_prev=None):
        """
        Sets current and previous simulation states

        Parameters
        ----------
        z_curr : (m, 1) float numpy array
            Current state of the reduced secondary motion sim
        p_curr : (12b, 1) float numpy array
            Current state of the rig parameters
        z_prev : (m, 1) float numpy array
            Previous state of the reduced secondary motion sim. If None, set to z_curr
        p_prev : (12b, 1) float numpy array
            Previous state of the rig parameter. If None, set to p_curr
        """
        z_curr = z_curr.copy()
        p_curr = p_curr.copy()

        if z_prev is None:
            z_prev = z_curr.copy()
        if p_prev is None:
            p_prev = p_curr.copy()

        super().__init__(z_curr, z_prev, p_curr, p_prev)


    '''
    Updates the simulation state
    Inputs:
    z - m x 1 reduced space coefficients for subspace sim
    p - 12b x 1 rig parameters
    '''
    def update(self, z, p):
        """
        Updates the simulation state

        Parameters
        ----------
        z : (m, 1) float numpy array
            Next state of the reduced secondary motion sim
        p : (12b, 1) float numpy array
            Next state of the rig parameters
        """
        self.z_prev = self.z_curr.copy()
        self.p_prev = self.p_curr.copy()

        self.z_curr = z.copy()
        self.p_curr = p.copy()


'''
Fast CD Simulation, implementation of https://www.dgp.toronto.edu/projects/fast_complementary_dynamics_site/
'''
class fast_cd_sim():
    """
    Fast Complementary Dynamics Simulation, implementation of https://www.dgp.toronto.edu/projects/fast_complementary_dynamics_site/
    """
    def __init__(self, V, T, B, l, J, mu=1e4, rho=1e3, h=1e-2, max_iters=30, threshold=1e-8,
                 read_cache=False, cache_dir="", Aeq=None, write_cache=False):
        """
        Initializes a Fast Complementary Dynamics Simulation.
        
        Parameters
        ----------
        V : (n, 3) float numpy array
            Vertex positions
        T : (F, 4) int numpy array
            Tet indices
        B : (3n, m) float numpy array
            Subspace matrix
        l : (F, 1) int numpy array
            Cluster labels
        J : (3n, 12b) float numpy array
            LBS rig jacobian
        mu : float 
            First lame parameter (default=1e5)
        rho : float
            Density (default=1)
        h : float
            Timestep (default=1e-2)
        max_iters : int
            Maximum number of iterations for the local-global solver (default=30)
        threshold : float
            Convergence threshold for local global solver (default=1e-8)
        read_cache : bool
            Whether to read simulation precomp from cache (default=False)
        cache_dir : str
            Directory to read/write cache from/to (default="")
        Aeq : (c, m) float numpy array
            Constraint matrix (default empty, untested yet)
        write_cache : bool
            Whether to write simulation precomp to cache (default=False) 
        """
        #These parameters need to be global member variables otherwise their memory is destroyed
        self.solver_params = fcd.local_global_solver_params(False, max_iters, threshold)
        if Aeq is None:
            self.Aeq = sp.sparse.csc_matrix((0, 0))
        self.Jsp = sp.sparse.csc_matrix(J)
        self.sim_params = fcd.fast_cd_arap_sim_params(V, T, B, l, self.Jsp, self.Aeq, mu, h, rho)

        write_cache = write_cache
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
        """
        Steps simulation state forward

        Parameters
        ----------
        p : (12b, 1) float numpy array
            Next state of the rig parameters
        state : fast_cd_state
            Current state of the simulation
        f_ext : (m, 1) float numpy array
            External force (default=0)
        bc : (c, 1) float numpy array
            Boundary constraints (default=None). Only valid if fast_cd_sim.Aeq is non-empty

        Returns
        -------
        z_next : (m, 1) float numpy array
            Next state of the reduced secondary motion sim
        """


        if f_ext is None:
            f_ext = np.zeros((state.z_curr.shape[0], 1))
        if bc is None:
            bc = np.array([[]], dtype=np.float64).T
        else:
            assert(bc.shape[0] == self.Aeq.shape[0])
        if z is None:
            z = state.z_curr
        assert(bc.shape[0] == self.Aeq.shape[0] and "Constraint rhs and matrix must have same number of rows")

        z = self.sim.step(z, p, state, f_ext, bc)[:, None]

        return z
