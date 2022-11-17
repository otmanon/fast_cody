import polyscope as ps
import numpy as np
import igl
import fast_cd_pyb as fcd
import scipy as sp
import os

name = "giraffe"
V = igl.read_dmat("./data/" + name + "/V.DMAT")
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
U = (igl.read_dmat("./data/" + name + "/U.DMAT")/so) - to;
T = igl.read_dmat("./data/" + name + "/T.DMAT").astype(np.int32)
F = igl.boundary_facets(T)
M = fcd.massmatrix(V, T)
M = sp.sparse.kron( sp.sparse.identity(3), M)
W = np.array([]);
J = sp.sparse.csc_matrix((V.shape[0]*3, 0))


num_modes = 9
[B_lbs, Ws, L] = fcd.get_modes(V, T, W, J,  "skinning", num_modes)
#B_lbs = sp.sparse.identity(int(V.shape[0]*3))

######### FIT rotations to LBS Subspace for bar ###########
A = B_lbs.T @ M @ B_lbs
rhs = B_lbs.T @ M @(U-V).flatten(order='F')
z = np.linalg.solve(A, rhs)
Urec_lbs =np.reshape(B_lbs @ z, (V.shape[0], 3), order="F")

# ############   FIT rotations to DISP Subspace forbar  #######
[B_disp, Ws, L] = fcd.get_modes(V, T, W, J,  "displacement", 12*num_modes)
A = B_disp.T @ M @ B_disp
rhs = B_disp.T @ M @ (U-V).flatten(order='F')
z = rhs
Urec_disp =np.reshape(B_disp @ z, (V.shape[0], 3), order="F")




## Once we have the modes, fit them to each displacement
ps.init()
#ps.register_volume_mesh(name="rest", vertices=V, tets=T)
ps.register_volume_mesh(name="ref", vertices=U, tets=T)
ps.register_volume_mesh(name="rec_lbs", vertices=V+Urec_lbs, tets=T)
ps.register_volume_mesh(name="rec_disp", vertices=V+Urec_disp, tets=T)

result_dir = "./results/fit_modes/" + name + "/"
try: 
    os.makedirs(result_dir) 
except OSError as error: 
    print(error) 

np.save(result_dir + "V.npy", V)
np.save(result_dir + "F.npy", F)
np.save(result_dir + "P.npy",  U )
np.save(result_dir + "P_lbs.npy", V + Urec_lbs  )
np.save(result_dir + "P_disp.npy", V + Urec_disp  )


ps.show()