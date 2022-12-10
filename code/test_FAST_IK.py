import fast_cd_pyb as fcd
import igl
import numpy as np
import polyscope as ps
msh_file = "./data/charizard/charizard.msh"
rig_dir = "./data/charizard/skeleton_rig_arms_legs/"
rig_file = rig_dir + "/skeleton_rig_arms_legs.json"

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
                    "./data/charizard/skeleton_rig_arms_legs/cache/", True);


bI = np.array([1])
solver_params = fcd.cd_arap_local_global_solver_params(True, 100, 1e-6)
sim = fcd.fast_ik_sim(V, T, W, sub.labels, bI, solver_params);

print("Done!")


ps.init()

mesh = ps.register_volume_mesh("mesh", V, T)

B = fcd.lbs_jacobian(V, W)
num_b = P0.shape[0]/4
p = np.zeros((0, 1))
z = np.zeros(B.shape[1])
state = fcd.cd_sim_state(z, z, p, p)
f_ext = np.zeros(z.shape);
step = 0
d = [1]

def callback():
    global step
    bc = step *  np.array(d)
    z_next = sim.step(z, p, state, f_ext, bc)
    u = B*z_next
    V_next = u.reshape((V.shape[0], 3))
    mesh.update_vertex_positions(V_next)
    step += 1
    pass

ps.set_user_callback(callback)
ps.show()
