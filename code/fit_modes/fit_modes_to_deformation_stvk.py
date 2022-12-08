import polyscope as ps
import numpy as np
import igl
import fast_cd_pyb as fcd
import scipy as sp
import os

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

[B_lbs, Ws, L] = fcd.get_modes(V, T, W, J,  "skinning", num_modes)
#B_lbs = sp.sparse.identity(int(V.shape[0]*3))

######### FIT rotations to LBS Subspace for bar ###########
A = B_lbs.T @ M @ B_lbs
rhs = B_lbs.T @ M @(Vd-V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_lbs =np.reshape(B_lbs @ z, (V.shape[0], 3), order="F")

# ############   FIT rotations to DISP Subspace forbar  #######
[B_disp, Ws, L] = fcd.get_modes(V, T, W, J,  "displacement", 12*num_modes)
rhs = B_disp.T @ M @ (Vd-V).flatten(order='F')
z = rhs
Urec_disp =np.reshape(B_disp @ z, (V.shape[0], 3), order="F")
## Once we have the modes, fit them to each displacement
ps.init()
ps.register_volume_mesh(name="rest", vertices=V, tets=T)
ps.register_volume_mesh(name="ref", vertices=Vd, tets=T)
ps.register_volume_mesh(name="rec_lbs", vertices=V+Urec_lbs, tets=T)
ps.register_volume_mesh(name="rec_disp", vertices=V+Urec_disp, tets=T)


try: 
    os.makedirs(result_dir) 
except OSError as error: 
    print(error) 

np.save(result_dir + "V.npy", V)
np.save(result_dir + "F.npy", F)
np.save(result_dir + "P.npy",  Vd )
np.save(result_dir + "P_lbs.npy", V + Urec_lbs  )
np.save(result_dir + "P_disp.npy", V + Urec_disp  )


ps.show()