import polyscope as ps
import fast_cd_pyb as fcd
import numpy as np
import igl 
from sklearn.cluster import KMeans
import scipy.sparse as sp
name = "charizard"
mesh_file = "./data/charizard/" + name + ".msh"
[V, T] = igl.read_msh(mesh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
F = igl.boundary_facets(T);

rigs = ["skeleton_rig"]

result_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/local_modes_visualization/charizard/"

for rig in rigs:
    rig_dir = result_dir + rig 
    weight_file = rig_dir + "/W.DMAT"
    W = igl.read_dmat(weight_file)
    success = np.save(rig_dir + "/W.npy", W);
    success = np.save(rig_dir + "/V.npy", V);
    success = np.save(rig_dir + "/F.npy", F);


