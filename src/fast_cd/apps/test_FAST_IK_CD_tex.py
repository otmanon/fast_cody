import fast_cd_pyb as fcd
import igl
import numpy as np
import scipy as sp
import polyscope as ps
msh_file = "./data/charizard/charizard.msh"
rig_dir = "./data/charizard/skeleton_rig_arms_legs/"
rig_file = rig_dir + "/skeleton_rig_arms_legs.json"

texture_obj = "./data/charizard/charizard_tex.obj"
texture_png = "./data/charizard/charizard_tex.png"

read_cache = False
write_cache = True


######## SIM PARAMS #########
num_clusters = 50
num_skinning_modes =15
mu = 20.0
lam = 0
h = 1e-2
do_inertia = True
cache_cd = "./results/fast_cd_ik_cache/cd/"
cache_ik = "./results/fast_cd_ik_cache/ik/"
########## LOAD RIG ##########
[V, F, T] = fcd.readMSH(msh_file)
[W, P0, pI, bl, Vs, Fs, rig_type] = fcd.read_rig_from_json(rig_file)
A = np.identity(4)[:3, :4]
if (rig_type == "surface"):
    W = fcd.surface_to_volume_weights(W, Vs, V, T)
    [P0, A] = fcd.fit_rig_to_mesh_surface(V, T, Vs, P0)
    fcd.write_volume_mesh = (W, P0, pI, bl, "volume", rig_dir + "./volume_skeleton_rig.json")
else:
    [P0, A] = fcd.fit_rig_to_mesh(V, Vs, P0)
J = fcd.lbs_jacobian(V, W) # rig jacobian

########## Load IK SIM ##########
num_clusters = 10;
sub = fcd.fast_ik_subspace(num_clusters, read_cache, "./debug/");
sub.init_with_cache(V, T, W, read_cache, write_cache,
                    cache_ik, True);
bI = np.array([1])
solver_params = fcd.local_global_solver_params(True, 10, 1e-3)
bI = np.array([])
S = fcd.selection_matrix(bI, V.shape[0], V.shape[1])
labels2 = np.copy(sub.labels)
sim_ik = fcd.fast_ik_sim(V, T, W,  labels2, S, solver_params);

p = np.zeros((W.shape[1] * 12, 1))
p = np.tile(np.identity(4)[:3, :4], W.shape[1]).reshape(p.shape)
state_ik = fcd.sim_state(p, p)

#repeat identity matrix for each column in W and flatten with order="F"
# p = np.tile(np.identity(4)[:3, :4], W.shape[1]).reshape((V.shape[0] * 12, 1), order="F")


sub_cd = fcd.fast_cd_subspace(num_skinning_modes, "cd_momentum_leak",
                              "skinning",   num_clusters, num_skinning_modes, True)
sub_cd.init_with_cache(V, T, J, read_cache, write_cache,
                       cache_cd, cache_cd, True, True)
solver_params = fcd.local_global_solver_params(False, 100, 1e-4)
labels = np.copy(sub_cd.labels)  # need to copy this, for some reason its not writeable otherwise
B = np.copy(sub_cd.B)
lam = 0.49
sim_params = fcd.fast_cd_corot_sim_params(V, T, B, labels, J,  mu, lam, h, do_inertia)
# sim_cd = fcd.fast_cd_arap_sim(cache_cd, sim_params, solver_params, False, write_cache)
sim_cd = fcd.fast_cd_corot_sim(sim_params, solver_params)

z = np.zeros((sub_cd.W.shape[1] * 12, 1));
# z = np.tile(np.identity(4)[:3, :4], sub_cd.W.shape[1]).reshape(z.shape)
state_cd = fcd.cd_sim_state(z, z, p, p)

# mesh = ps.register_volume_mesh("mesh", V, T)

# B = fcd.lbs_jacobian(V, W)
# num_b = P0.shape[0]/4
# p = np.zeros((0, 1))
# z = np.zeros(B.shape[1])
# state = fcd.sim_state(z, z)
# f_ext = np.zeros(z.shape);
# step = 0
# d = [1]
f_ext_cd = np.zeros(z.shape)
f_ext_ik = np.zeros(p.shape)

bc_ik = np.array([])
bc_cd = np.array([])

########## INIT VIEWER ##########
v = fcd.fast_cd_viewer_vertex_selector()
F = igl.boundary_facets(T)
[Vf, TC, N, Ff, FTC, FN] = fcd.readOBJ_tex(texture_obj);
P = sp.sparse.kron(sp.sparse.identity(3), fcd.prolongation(Vf, V, T))
v.set_mesh(Vf, Ff, 0)
v.set_show_lines(False, 0)
# v.set_mesh(V, F, 0)
v.set_texture(texture_png, TC, FTC, 0)
# v.invert_normals(True, 0)
v.set_face_based(False, 0)
Vc = V
V_next = Vf
def callback():
    global bc_ik, V_next, Vc
    [C, CI, new_handles] = v.query_new_handles_on_mesh(Vc, F)
    if (new_handles):
        S = fcd.selection_matrix(CI, V.shape[0], V.shape[1])
        sim_ik.set_equality_constraint(S)
    bc_ik = np.reshape(C, (C.shape[0] * C.shape[1]), order="F")
    # p_next_ik = p
    p_next_ik = sim_ik.step(p, state_ik, f_ext_ik, bc_ik)
    state_ik.update(p_next_ik)

    z_next_cd = sim_cd.step(z, p_next_ik, state_cd, f_ext_cd, bc_cd)
    state_cd.update(z_next_cd, p_next_ik)

    x = np.reshape(V, (V.shape[0] * V.shape[1]), order="F")
    Vc =  (J @ p_next_ik + B@z_next_cd).reshape(V.shape, order="F")
    # Vc = V + (B@z_next_cd).reshape(V.shape, order="F")
    u =  B@z_next_cd  + J@p_next_ik - x

    uf = P@u;
    V_next = uf.reshape((Vf.shape[0], 3), order="F") + Vf
    v.set_vertices(V_next, 0)
    v.compute_normals(0)

    return


v.set_pre_draw_callback(callback)

v.launch()