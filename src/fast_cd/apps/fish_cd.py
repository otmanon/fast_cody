import fast_cd_pyb as fcdp
import numpy as np
import scipy as sp
import igl
import fast_cd as fcd
class fish_cd():
    def __init__(self, result_dir=None):


        self.msh_file = fcd.get_data('cd_fish.msh')
        self.texture_obj = fcd.get_data('cd_fish_tex.obj')
        self.texture_png = fcd.get_data('cd_fish_tex.png')
        # rig_file = "./data/" + name + "/rigs/skeleton_rig/skeleton_rig.json"
        self.result_dir = ''

        read_cache = False
        write_cache = False
        num_modes = 8
        mode_type = "skinning"
        num_clusters = 50
        mu = 1e4
        rho=1e3
        h = 1e-2



        [V, F, T] = fcdp.readMSH(self.msh_file)
        F = igl.boundary_facets(T)
        self.F = F
        self.T = T
        [V, so, to] = fcdp.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))
        self.V = V
        self.so = so
        self.to = to
        self.Wp = np.ones((V.shape[0], 1));
        self.bI = 0 # bone index in rig that we care about

        constraint_enforcement="optimal"
        J = fcd.lbs_jacobian(V, self.Wp)
        self.J = J;
        C = fcd.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fcd.lbs_weight_space_constraint(V, C)
        [B, l, self.Ws] = fcd.skinning_subspace(V, T, num_modes, num_clusters, C=C2, read_cache=read_cache,
                                           cache_dir=None, constraint_enforcement=constraint_enforcement);
        self.l= np.copy(l)  # need to copy this, for some reason its not writeable otherwise
        self.B = np.copy(B)
        self.sim = fcd.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=1e-2, cache_dir="./results", read_cache=read_cache)
