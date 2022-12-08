import polyscope as ps
import numpy as np
import igl
import fast_cd_pyb as fcd
import scipy as sp
import os
from build_vertex_rotation_matrix import build_vertex_rotation_matrix

from sklearn.cluster import KMeans
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
# M = fcd.massmatrix(Vd, T)
# M = sp.sparse.kron( sp.sparse.identity(3), M)
# W = np.array([]);
# J = sp.sparse.csc_matrix((V.shape[0]*3, 0))


# # widthwise to a bin
num_clusters = 1000
# # group vertices according to their x coordinate
# # min_x = np.min(V[:,0])
# # width = np.max(V[:,0]) - min_x
# # clusters = np.floor((V[:,0] - min_x) / width * num_clusters).astype(int)
# # clusters = np.minimum(clusters, num_clusters-1);
# kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(V)
# clusters = kmeans.labels_

# # # ############   FIT rotations to DISP Subspace forbar  #######
# [B_disp, Ws, L] = fcd.get_modes(V, T, W, J,  "displacement", 12*num_modes)
# # #build some vertex clusters very trivially, for ten clusters, each vertex is assigned
# #initialize zero sparse full space rotation matrix
# Rv = sp.sparse.csc_matrix((V.shape[0]*3, V.shape[0]*3))
# for c in range(0, num_clusters):
#     Ind = np.where(clusters == c)[0]
#     P = Vd[Ind, :] - Vd[Ind, :].mean(0);
#     P0 = V[Ind, :] - V[Ind, :].mean(0);
#     C = P.T @ P0;
#     # # #fit rotation to C using igl
#     [R, S] = igl.polar_dec(C);
#    # R =np.array([[0, 0, 1.0], [0.0, 1, 0.0], [ -1, 0.0, 0.0]]);
#     Rv = Rv + build_vertex_rotation_matrix(R, Ind, V.shape[0], V.shape[1])
#     #clusters = np.arange(0, V.shape[0]); #these are vertex clusters
    
# # R =np.array([[0, 0, 1.0], [0.0, 1, 0.0], [ -1, 0.0, 0.0]]);
# # Rv = build_vertex_rotation_matrix(R, np.arange(0, V.shape[0]), V.shape[0], V.shape[1])
# # v = np.reshape(V, V.shape[0]*V.shape[1], order="F")
# # U =np.reshape(Rv @ v, (V.shape[0], 3), order="F")
# #U = (R @ V.T).T
# # # fcd.compute_clusters(T, B_disp, L, num_clusters, num_clustering_features)
# rhs = B_disp.T @ Rv.T @ M @ (Vd-V).flatten(order='F')

# D =np.reshape(Rv @ B_disp @ rhs, (V.shape[0], 3), order="F")
# U = D + V;
# # ## Once we have the modes, fit them to each displacement
# ps.init()
# mesh2 = ps.register_volume_mesh(name="ref", vertices=V, tets=T)
# mesh = ps.register_volume_mesh(name="rec", vertices=U, tets=T)

tet_barycenters = igl.barycenter(V, F)

kmeans_tets = KMeans(n_clusters=800, random_state=0).fit(tet_barycenters)
clusters_tets = kmeans_tets.labels_
# mesh.add_scalar_quantity(name="clusters", values=clusters_tets, enabled=True, 
# defined_on='cells', cmap="rainbow")

# try: 
#     os.makedirs(result_dir) 
# except OSError as error: 
#     print(error) 

# # np.save(result_dir + "V.npy", V)
# # np.save(result_dir + "F.npy", F)
# np.save(result_dir + "P_disp_1000_floating_frames.npy", U )

# [F, FiT] = igl.boundary_facets(T)
np.save(result_dir + "face_clusters_1000.npy", clusters_tets)


ps.show()