import polyscope as ps
import numpy as np
import igl
import fast_cd_pyb as fcd
import scipy as sp
import os
V = igl.read_dmat("./data/raptor/rotate_down/V.DMAT")
U_up = igl.read_dmat("./data/raptor/rotate_up/U.DMAT")
U_down = igl.read_dmat("./data/raptor/rotate_down/U.DMAT")
T = igl.read_dmat("./data/raptor/rotate_down/T.DMAT").astype(np.int32)
F = igl.boundary_facets(T)
M = fcd.massmatrix(V, T)
M = sp.sparse.kron( sp.sparse.identity(3), M)
W = np.array([]);
J = sp.sparse.csc_matrix((V.shape[0]*3, 0))


num_modes = 10
[B_lbs, Ws, L] = fcd.get_modes(V, T, W, J,  "skinning", num_modes)


######### FIT rotations to LBS Subspace for Raptor ###########
A = B_lbs.T @ M @ B_lbs
rhs = B_lbs.T @ M @(U_down - V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_down_lbs =np.reshape(B_lbs @ z, (V.shape[0], 3), order="F")

rhs = B_lbs.T @ M @(U_up -V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_up_lbs=np.reshape(B_lbs @ z, (V.shape[0], 3), order="F")

############   FIT rotations to DISP Subspace for Raptor   #######
[B_disp, Ws, L] = fcd.get_modes(V, T, W, J,  "displacement", 12*num_modes)
A = B_disp.T @ M @ B_disp
rhs = B_disp.T @ M @ (U_down-V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_down_disp =np.reshape(B_disp @ z, (V.shape[0], 3), order="F")

rhs = B_disp.T @ M @ (U_up -V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_up_disp =np.reshape(B_disp @ z, (V.shape[0], 3), order="F")


## Once we have the modes, fit them to each displacement
ps.init()
ps.register_volume_mesh(name="rest", vertices=V, tets=T)
ps.register_volume_mesh(name="down_ref", vertices=U_down, tets=T)
ps.register_volume_mesh(name="down_rec_lbs", vertices=V+Urec_down_lbs, tets=T)
ps.register_volume_mesh(name="down_rec_disp", vertices=V+Urec_down_disp, tets=T)
ps.register_volume_mesh(name="up_ref", vertices=U_up, tets=T)
ps.register_volume_mesh(name="up_rec_lbs", vertices=V+Urec_up_lbs, tets=T)
ps.register_volume_mesh(name="up_rec_disp", vertices=V+Urec_up_disp, tets=T)

result_dir = "./results/fit_modes/raptor/"
try: 
    os.makedirs(result_dir) 
except OSError as error: 
    print(error) 

np.save(result_dir + "V.npy", V)
np.save(result_dir + "F.npy", F)
np.save(result_dir + "P_up.npy",  U_up  )
np.save(result_dir + "P_down.npy", U_down  )
np.save(result_dir + "P_up_lbs.npy", V + Urec_up_lbs  )
np.save(result_dir + "P_down_lbs.npy", V + Urec_down_lbs  )
np.save(result_dir + "P_up_disp.npy", V + Urec_up_disp  )
np.save(result_dir + "P_down_disp.npy", V + Urec_down_disp  )
ps.show()