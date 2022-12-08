import polyscope as ps
import numpy as np
import igl
import fast_cd_pyb as fcd
import scipy as sp
import os
from build_vertex_rotation_matrix import build_vertex_rotation_matrix
name = "bar"
frame = 8
num_modes = 4
[V, n, T] = fcd.readMSH("C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/data/raw_data/" + name + "/" + name + ".msh")
deform_path = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/skeleton_rig_cd_animation/" + name + "/mesh_recording/" + str(frame).zfill(4) + ".obj"
result_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/mode_fitting/" + name + "/npy/"  + str(frame) +"/"
[Vd, Fd] = fcd.readOBJ(deform_path)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
#[Vd, so, to] = fcd.scale_and_center_geometry(Vd, 1, np.array([[0, 0,  0.]]))

F = igl.boundary_facets(T)
M = fcd.massmatrix(Vd, T)
M = sp.sparse.kron( sp.sparse.identity(3), M)
W = np.array([]);
J = sp.sparse.csc_matrix((V.shape[0]*3, 0))


P = Vd - Vd.mean(0);
P0 = V - V.mean(0);
C = P.T @ P0;
#fit rotation to C using igl
[R, S] = igl.polar_dec(C);
clusters = np.arange(0, V.shape[0]); #these are vertex clusters
Rv = build_vertex_rotation_matrix(R, clusters, V.shape[0], V.shape[1])
# ############   FIT rotations to DISP Subspace forbar  #######
[B_disp, Ws, L] = fcd.get_modes(V, T, W, J,  "displacement", 12*num_modes)
# fcd.compute_clusters(T, B_disp, L, num_clusters, num_clustering_features)
rhs = B_disp.T @ Rv.T @ M @ (Vd-V).flatten(order='F')
z = rhs
U =np.reshape(Rv @ B_disp @ z, (V.shape[0], 3), order="F")
## Once we have the modes, fit them to each displacement
ps.init()
ps.register_volume_mesh(name="ref", vertices=Vd, tets=T)
ps.register_volume_mesh(name="rec", vertices=V+U, tets=T)


try: 
    os.makedirs(result_dir) 
except OSError as error: 
    print(error) 

np.save(result_dir + "V.npy", V)
np.save(result_dir + "F.npy", F)
np.save(result_dir + "P_disp_floating.npy", V + U )


ps.show()