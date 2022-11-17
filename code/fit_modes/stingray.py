import polyscope as ps
import numpy as np
import igl
import fast_cd_pyb as fcd
import scipy as sp
import os
scale = 1
V = scale*igl.read_dmat("./data/stingray/flap_down/V.DMAT")

[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
U_up = scale * (igl.read_dmat("./data/stingray/flap_up/U.DMAT")/so) - to;
U_down = scale* (igl.read_dmat("./data/stingray/flap_down/U.DMAT")/so) - to;
T = igl.read_dmat("./data/stingray/flap_down/T.DMAT").astype(np.int32)
F = igl.boundary_facets(T)
M = fcd.massmatrix(V, T)
M = sp.sparse.kron( sp.sparse.identity(3), M)
W = np.array([]);
J = sp.sparse.csc_matrix((V.shape[0]*3, 0))


num_modes = 5
[B_lbs, Ws, L] = fcd.get_modes(V, T, W, J,  "skinning", num_modes)


######### FIT rotations to LBS Subspace for stingray ###########
A = B_lbs.T @ M @ B_lbs; # + 1e-6 * np.identity(B_lbs.shape[1])
rhs = B_lbs.T @ M @(U_down-V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_down_lbs =np.reshape(B_lbs @ z, (V.shape[0], 3), order="F")

rhs = B_lbs.T @ M @(U_up-V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_up_lbs=np.reshape(B_lbs @ z, (V.shape[0], 3), order="F")

############   FIT rotations to DISP Subspace for stingray   #######
[B_disp, Ws, L] = fcd.get_modes(V, T, W, J,  "displacement", 12*num_modes)

z = B_disp.T @ M @(U_down-V).flatten(order='F')
Urec_down_disp =np.reshape(B_disp @ z, (V.shape[0], 3), order="F")

rhs = B_disp.T @ M @ (U_up-V).flatten(order='F')
z = rhs
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

result_dir = "./results/fit_modes/stingray/"
try: 
    os.makedirs(result_dir) 
except OSError as error: 
    print(error) 

np.save(result_dir + "V.npy", V)
np.save(result_dir + "F.npy", F)
np.save(result_dir + "P_up.npy",  U_up  )
np.save(result_dir + "P_down.npy",  U_down  )
np.save(result_dir + "P_up_lbs.npy", V + Urec_up_lbs  )
np.save(result_dir + "P_down_lbs.npy", V + Urec_down_lbs  )
np.save(result_dir + "P_up_disp.npy", V + Urec_up_disp  )
np.save(result_dir + "P_down_disp.npy", V + Urec_down_disp  )
ps.show()