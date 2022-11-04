import numpy as np 
import scipy as sp
import igl
import fast_cd_pyb as fcd 
import json
import os.path

write_cache = True
read_cache = True

name = "bulldog"
mesh_file = "./data/" + name + ".msh"
cache_dir = "./cache/" + name + "/"
meta_file = cache_dir + "/meta.json"
[V, F, T] = fcd.readMSH(mesh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))

F = igl.boundary_facets(T);


params = fcd.fast_cd_sim_params();
viewer = fcd.fast_cd_viewer()

viewer.set_mesh(V, F, 0)
viewer.invert_normals(True, 0)

color = np.array([144, 210, 236])/255.0
viewer.set_color(color, 0)

W = np.ones((V.shape[0], 1))
J = fcd.lbs_jacobian(V, W)

##### SIMULATION PARAMETERS
num_modes = 30
num_clusters = 100
num_clustering_features = 10
mu = 10
h = 1e-2
lam = 0
mode_type = "skinning"
do_inertia = True

cache_params = {"num_modes" : num_modes, 
"num_clusters" : num_clusters, 
"mu" : mu, "h" : h, 
"lambda" : lam, 
"mode_type" : mode_type, 
"do_inertia" : do_inertia}

solver_params = fcd.cd_arap_local_global_solver_params(True, 10, 1e-3)
sim = fcd.fast_cd_arap_sim();
well_read = False
if read_cache and os.path.isfile(meta_file) :
    f = open(meta_file)
    data = json.load(f)
    if data == cache_params:
        sim  = fcd.fast_cd_arap_sim(cache_dir, solver_params)
        B2 = sim.params.B.copy();
        labels2  =  sim.params.labels.copy()
        sim.params = fcd.fast_cd_sim_params(V, T, B2, labels2, J, mu, lam, h, do_inertia) 
        well_read = True
if not well_read:
    [B, Ws, L] = fcd.get_modes(V, T, W, J, mode_type, num_modes)
    [labels, C] = fcd.compute_clusters(T, B, L, num_clusters, num_clustering_features)
    sim_params = fcd.fast_cd_sim_params(V, T, B, labels, J, mu, lam, h, do_inertia) 
    sim = fcd.fast_cd_arap_sim(sim_params, solver_params)
    if (write_cache):
        sim.save(cache_dir);
        with open(meta_file, 'w') as outfile:
                json.dump(cache_params, outfile, indent=2)


# set sim state 
z0 = np.zeros((num_modes*12, 1))
T0 = np.identity(4).astype( dtype=np.float32, order="F");
p0 = T0[0:3, :].reshape( (12, 1))
st = fcd.cd_sim_state(z0, z0, p0, p0)

# momentum leaking matrix
M = fcd.massmatrix(V, T)
D = fcd.momentum_leaking_matrix(V, T)
md = np.tile((M@ D).diagonal(), (3))
MD = sp.sparse.diags(md)
BMDJ = sim.params.B.T @  MD @ J


def callback():
     global J, B, T0, sim, st
     p = T0[0:3, :].reshape( (12, 1))
     f_ext = sim.params.invh2 * BMDJ @ (2.0 * st.p_curr.reshape((12, 1), order="F") - st.p_prev.reshape((12, 1), order="F") - p)
     bc = np.array([[]], dtype=np.float64).T
     #initial guess... 
     z = st.z_curr   
     z = sim.step(z, p,  st, f_ext, bc).reshape((12*num_modes, 1), order="F")
     #print(z)
     st.update(z, p);
     # print(np.linalg.norm(B@z));
     U = np.reshape(J@p + sim.params.B@z, (int(J.shape[0]/3), 3), order="F")
     viewer.set_vertices(U, 0)

def guizmo_callback(A):
    global T0
    T0 = A

viewer.init_guizmo(True, T0, guizmo_callback, "translate")
viewer.set_pre_draw_callback(callback)
viewer.launch()