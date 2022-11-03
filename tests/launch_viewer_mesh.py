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
viewer.launch()