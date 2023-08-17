import fast_cd_pyb as fcd
import igl
import numpy as np
import scipy as sp
import polyscope as ps
# msh_file = "./data/charizard/charizard.msh"
# rig_dir = "./data/charizard/skeleton_rig_arms_legs/"
# rig_file = rig_dir + "/skeleton_rig_arms_legs.json"

# msh_file = "./data/king_ghidorah/king_g.msh"
# rig_dir = "./data/king_ghidorah/skeleton_rig/"
# rig_file = rig_dir + "/skeleton_rig.json"
# cache_ik = "./results/fast_ik_cache/king_ghidorah/ik/"
[V, F, T] = fcd.readMSH(msh_file)
[W, P0, pI, bl, Vs, Fs, rig_type] = fcd.read_rig_from_json(rig_file)
A = np.identity(4)[:3, :4]
if (rig_type == "surface"):
    W = fcd.surface_to_volume_weights(W, Vs, V, T)
    [P0, A] = fcd.fit_rig_to_mesh_surface(V, T, Vs, P0)
    fcd.write_volume_mesh = (W, P0, pI, bl, "volume", rig_dir + "./volume_skeleton_rig.json")
else:
    [P0, A] = fcd.fit_rig_to_mesh(V, Vs, P0)

num_clusters = 10;
sub = fcd.fast_ik_subspace(num_clusters, False, "./debug/");


sub.init_with_cache(V, T, W, True, True,
                    cache_ik, True);


bI = np.array([1])
solver_params = fcd.cd_arap_local_global_solver_params(True, 10, 1e-3)
bI = np.array([])
S = fcd.selection_matrix(bI, V.shape[0], V.shape[1])
labels = np.copy(sub.labels)
sim = fcd.fast_ik_sim(V, T, W,  labels, S, solver_params);

z = np.zeros((W.shape[1] * 12, 1))
z = np.tile(np.identity(4)[:3, :4], W.shape[1]).reshape(z.shape)
state = fcd.sim_state(z, z)

print("Done!")



# mesh = ps.register_volume_mesh("mesh", V, T)

# B = fcd.lbs_jacobian(V, W)
# num_b = P0.shape[0]/4
# p = np.zeros((0, 1))
# z = np.zeros(B.shape[1])
# state = fcd.sim_state(z, z)
# f_ext = np.zeros(z.shape);
# step = 0
# d = [1]
f_ext = np.zeros(z.shape)
J = fcd.lbs_jacobian(V, W)
bc = np.array([])

v = fcd.fast_cd_viewer_vertex_selector()
F = igl.boundary_facets(T)

v.set_mesh(V, F, 0)
v.set_show_lines(False, 0)
v.set_face_based(False, 0)
v.invert_normals(True, 0)

X = V
def callback():
    global bc, X
    [C, CI, new_handles] = v.query_new_handles_on_mesh(X, F)
    if (new_handles):
        S = fcd.selection_matrix(CI, V.shape[0], V.shape[1])
        sim.set_equality_constraint(S)
    bc = np.reshape(C, (C.shape[0] * C.shape[1]), order="F")
    #
    # # global step
    # # bc = step *  np.array(d)
    z_next = sim.step(z,  state, f_ext, bc)

    X = (J @ z_next).reshape(V.shape, order="F")

    v.set_vertices(X, 0)
    v.compute_normals(0)

    return


v.set_pre_draw_callback(callback)

v.launch()