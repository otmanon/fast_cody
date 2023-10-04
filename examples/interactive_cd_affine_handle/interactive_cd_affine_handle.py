import numpy as np 
import scipy as sp
import igl
import fast_cd_pyb as fcd
import fast_cd as fc
import json
import os.path

write_cache = True
read_cache = False

## parameters
num_skinning_modes = 16
num_clusters = 400
mu = 10
lam = 0
h = 1e-2
do_inertia = True
name = "bulldog"
mesh_file = "../data/" + name + ".msh"
cache_cd = "./cache/" + name + "/"
meta_file = cache_cd + "/meta.json"
[V, F, T] = fcd.readMSH(mesh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))

F = igl.boundary_facets(T);


W = np.ones((V.shape[0], 1))
J = fc.lbs_jacobian(V, W)

result_dir = '../../examples/interactive_cd_affine_handle/results/'
cache_dir = os.path.join(result_dir, 'cache/')
# os.makedirs(cache_dir)
modes_dir =  cache_dir
clusters_dir = cache_dir
mode_type = 'skinning'
num_modes = 10
num_clusters = 100


# TODO: make fast_cd_subspace take in B, and l
M = sp.sparse.kron(sp.sparse.identity(3), \
                   igl.massmatrix(V, T))

C = fc.lbs_weight_space_constraint(V,  M @ J, M=igl.massmatrix(V, T))
[B, l, W] = fc.skinning_subspace(V, T, 10, 100, C=C);
sub_cd = fcd.fast_cd_subspace(modes_dir, clusters_dir, mode_type, num_modes, num_clusters)

sub_cd = fcd.fast_cd_subspace(B, l, W);


solver_params = fcd.cd_arap_local_global_solver_params(True, 10, 1e-4)
labels = np.copy(sub_cd.labels)  # need to copy this, for some reason its not writeable otherwise
B = np.copy(sub_cd.B)
sim_params = fcd.fast_cd_sim_params(V, T, B, labels, J,  mu, lam, h, do_inertia, "none")
sim_cd = fcd.fast_cd_arap_sim(cache_cd, sim_params, solver_params, read_cache, write_cache)

viewer = fcd.fast_cd_viewer()

viewer.set_mesh(V, F, 0)
viewer.invert_normals(True, 0)

color = np.array([144, 210, 236])/255.0
viewer.set_color(color, 0)

# set sim state 
z0 = np.zeros((num_skinning_modes*12, 1))
T0 = np.identity(4).astype( dtype=np.float32, order="F");
p0 = T0[0:3, :].reshape( (12, 1))
st = fcd.cd_sim_state(z0, z0, p0, p0)

# momentum leaking matrix
M = fcd.massmatrix(V, T)
D = fcd.momentum_leaking_matrix(V, T)
md = np.tile((M@ D).diagonal(), (3))
MD = sp.sparse.diags(md)
f_ext = np.zeros((z0.shape[0], 1))
bc = np.array([[]], dtype=np.float64).T
def callback():
     global J, B, T0, sim, st
     p = T0[0:3, :].reshape( (12, 1))

     #initial guess... 
     z = st.z_curr   
     z = sim_cd.step(z, p,  st, f_ext, bc).reshape((12*num_skinning_modes, 1), order="F")
     #print(z)
     st.update(z, p);
     # print(np.linalg.norm(B@z));
     U = np.reshape(J@p + sim_cd.params().B@z, (int(J.shape[0]/3), 3), order="F")
     viewer.set_vertices(U, 0)

def guizmo_callback(A):
    global T0
    T0 = A

viewer.init_guizmo(True, T0, guizmo_callback, "translate")
viewer.set_pre_draw_callback(callback)
viewer.launch()