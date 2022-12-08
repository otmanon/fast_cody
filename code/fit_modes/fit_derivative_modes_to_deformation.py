import polyscope as ps
import numpy as np
import igl
import fast_cd_pyb as fcd
import scipy as sp
import os

name = "spoon"
frame = 0
num_primary_modes = 5
num_secondary_modes = 31
[V, n, T] = fcd.readMSH("C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/data/raw_data/" + name + "/" + name + ".msh")
deform_path = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/skeleton_rig_cd_animation/" + name + "/mesh_recording/" + str(frame).zfill(4) + ".obj"
primary_mode_dir = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/data/raw_data/" + name + "/B_stvk.DMAT";
secondary_mode_dir = mode_dir = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/data/raw_data/" + name + "/B_stvk_derivs.DMAT";
result_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/mode_fitting/" + name + "/npy/"  + str(frame) +"/"
[Vd, Fd] = fcd.readOBJ(deform_path)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
#[Vd, so, to] = fcd.scale_and_center_geometry(Vd, 1, np.array([[0, 0,  0.]]))
#read primary modes
B_disp_primary = igl.read_dmat(primary_mode_dir)[:, :num_primary_modes]
#read secondary modes
B_disp_secondary = igl.read_dmat(secondary_mode_dir)[:,:num_secondary_modes]
B_disp = np.concatenate((B_disp_primary, B_disp_secondary), axis=1)

F = igl.boundary_facets(T)
M = fcd.massmatrix(Vd, T)
M = sp.sparse.kron( sp.sparse.identity(3), M)
W = np.array([]);
J = sp.sparse.csc_matrix((V.shape[0]*3, 0))


# ############   FIT rotations to DISP Subspace forbar  #######
# [B_disp, Ws, L] = fcd.get_modes(V, T, W, J,  "displacement", 12*num_modes)
A = B_disp.T @ M @ B_disp
rhs = B_disp.T @ M @ (Vd-V).flatten(order='F')
z =  np.linalg.solve(A, rhs)
Urec_disp =np.reshape(B_disp @ z, (V.shape[0], 3), order="F")
## Once we have the modes, fit them to each displacement
ps.init()
ps.register_volume_mesh(name="rest", vertices=V, tets=T)
ps.register_volume_mesh(name="ref", vertices=Vd, tets=T)
ps.register_volume_mesh(name="rec_disp_derivs", vertices=V+Urec_disp, tets=T)


try: 
    os.makedirs(result_dir) 
except OSError as error: 
    print(error) 

np.save(result_dir + "V.npy", V)
np.save(result_dir + "F.npy", F)
np.save(result_dir + "P.npy",  Vd )
np.save(result_dir + "P_disp_derivs.npy", V + Urec_disp  )


ps.show()