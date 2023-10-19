import polyscope as ps
import fast_cd_pyb as fcd
import numpy as np
import igl 
from sklearn.cluster import KMeans
import scipy.sparse as sp
name = "charizard"
mesh_file = "./data/" + name + ".msh"

cache_dir = "./cache/" + name + "/"
meta_file = cache_dir + "/meta.json"
[V, F, T] = fcd.readMSH(mesh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
F = igl.boundary_facets(T);

W = np.ones((V.shape[0], 1))
J = fcd.lbs_jacobian(V, W)

mode_type = "skinning"
num_modes = 10;
num_clusters = 100;
num_clustering_features = 10;
[B, Ws, L] = fcd.get_modes(V, T, W, J, mode_type, num_modes)

success = np.save("./data/" + name + "_weights", Ws);
success = np.save("./data/" + name + "_verts", V);
success = np.save("./data/" + name + "_faces", F);


