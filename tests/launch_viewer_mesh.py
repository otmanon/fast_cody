import numpy as np 
import scipy as sp
import igl
import fast_cd_pyb as fcd 


[V, F, T] = fcd.readMSH("./data/raptor.msh")
F = igl.boundary_facets(T);

f = fcd.fast_cd_sim_params();
f.X = V;
f.T = T;
#A= fcd.double_matrix(V);

params = fcd.fast_cd_sim_params();
viewer = fcd.fast_cd_viewer()

viewer.set_mesh(V, F, 0)
# U1 = V + 10
# U2 = V - 10
# id1 = viewer.add_mesh( U1, F )
# id2 = viewer.add_mesh( U2, F )
viewer.invert_normals(True, 0)
 # viewer.invert_normals(True, 1)
# viewer.invert_normals(True, 2)
color = np.array([144, 210, 236])/255.0
# color1 = np.array([0, 1, 0])
# color2 = np.array([0, 0, 1])
viewer.set_color(color, 0)



W = np.ones((V.shape[0], 1))
J = fcd.lbs_jacobian(V, W)


num_modes = 30

num_clusters = 100;
num_clustering_features = 10;

[B, Ws, L] = fcd.get_modes(V, T, W, J, "skinning", num_modes)
[labels, C] = fcd.compute_clusters(T, B, L, num_clusters, num_clustering_features)

solver_params = fcd.cd_arap_local_global_solver_params(True, 10, 1e-3)
sim_params = fcd.fast_cd_sim_params(V, T, B, labels, J, 100.0, 0.0, 1e-2, True) 

sim = fcd.fast_cd_arap_sim(sim_params, solver_params)
z0 = np.zeros((num_modes*12, 1))

T0 = np.identity(4).astype( dtype=np.float32, order="F");
p0 = T0[0:3, :].reshape( (12, 1))

st = fcd.cd_sim_state(z0, z0, p0, p0)

# momentum leaking matrix
M = fcd.massmatrix(V, T)
D = fcd.momentum_leaking_matrix(V, T)
md = np.tile((M@ D).diagonal(), (3))
MD = sp.sparse.diags(md)
BMDJ = B.T @  MD @ J


def callback():
     global J, B, T0, sim, st
     p = T0[0:3, :].reshape( (12, 1))
     f_ext = sim_params.invh2 * BMDJ @ (2.0 * st.p_curr.reshape((12, 1), order="F") - st.p_prev.reshape((12, 1), order="F") - p)
     bc = np.array([[]], dtype=np.float64).T
     #initial guess... 
     z = st.z_curr   
     z = sim.step(z, p,  st, f_ext, bc).reshape((12*num_modes, 1), order="F")
     #print(z)
     st.update(z, p);
     # print(np.linalg.norm(B@z));
     U = np.reshape(J@p + B@z, (int(J.shape[0]/3), 3), order="F")
     viewer.set_vertices(U, 0)

def guizmo_callback(A):
    global T0
    T0 = A

viewer.init_guizmo(True, T0, guizmo_callback, "translate")
viewer.set_pre_draw_callback(callback)
viewer.launch()