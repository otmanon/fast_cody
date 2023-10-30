import polyscope as ps
import numpy as np
import igl
import fast_cody as fcd
import fast_cd_pyb as fcdp
name = "stingray"


frame = 20
num_modes = 3


msh_file = "./data/stingray.msh"
rig_dir = "./data/rigs/skeleton_rig/"
rig_json = rig_dir + "./skeleton_rig.json"
anim_json = "./data/rigs/skeleton_rig/anim/flap.json"


[V, F,  T] = fcd.readMSH(msh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))


[W, P0,pI, bl, Vs, Fs, rig_type ] = fcd.read_rig_from_json(rig_json)
A = np.identity(4)[:3, :4]

if (rig_type == "surface"):
    W = fcd.surface_to_volume_weights(W, Vs, V, T)
    a = fcd.fit_rig_to_mesh_surface(V, T, Vs, P0)
    fcd.write_volume_mesh = (W, P0, pI, bl, "volume", rig_dir + "./volume_skeleton_rig.json")
else:
    [P0, A] = fcd.fit_rig_to_mesh(V, Vs, P0)
#read rig animations
anim_P = fcd.read_anim_from_json(anim_json)
anim_P = fcd.transform_rig_parameters_anim(anim_P, A)
p0 = np.reshape(P0, (P0.shape[0]*3, 1), order="F")
anim_P = fcd.world_to_rel_rig_anim(anim_P, p0)
J = fcd.lbs_jacobian(V, W)

ps.init()

F = igl.boundary_facets(T)
mesh = ps.register_volume_mesh(name="rest", vertices=V, tets=T)

step = 0
num_frames = anim_P.shape[1]
def callback():
    global step
    u = J @ anim_P[:, step%num_frames]  #world space rig parameters
    U = np.reshape(u, (u.shape[0]//3, 3), order="F")
    mesh.update_vertex_positions(U)
    step += 1


ps.set_user_callback(callback)
ps.show()
