import numpy as np 
import scipy as sp
import igl
import fast_cd_pyb as fcd 
[V, F, T] = fcd.readMSH("../fast_cd_cpp/data/raw_data/bulldog/bulldog.msh")
F = igl.boundary_facets(T);


#A= fcd.double_matrix(V);

params = fcd.fast_cd_sim_params();
viewer = fcd.fast_cd_viewer()

viewer.set_mesh(V, F, 0)
U1 = V + 10
U2 = V - 10
id1 = viewer.add_mesh( U1, F )
id2 = viewer.add_mesh( U2, F )
viewer.invert_normals(True, 0)
viewer.invert_normals(True, 1)
viewer.invert_normals(True, 2)
color = np.array([1, 0, 0])
color1 = np.array([0, 1, 0])
color2 = np.array([0, 0, 1])
viewer.set_color(color, 0)
viewer.set_color(color1, id1)
viewer.set_color(color2, id2)


T0 = np.identity(4).astype( dtype=np.float32, order="F");

def callback():
     global V, T0
     t = T0[0:3, 3]
     U = V + t.T
     viewer.set_vertices(U, 0)
     
def guizmo_callback(A):
    global T0
    T0 = A

viewer.init_guizmo(True, T0, guizmo_callback, "translate")
viewer.set_pre_draw_callback(callback)
viewer.launch()