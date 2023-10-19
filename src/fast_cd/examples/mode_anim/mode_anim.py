import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import os
import numpy as np
import polyscope as ps
import fast_cd_pyb as fcd
import igl
import scipy as sp
import json
cwd = os.getcwd()


# User params
name = "charizard"
rig_names = ["skeleton_rig_arms_legs"] #"null", "single_bone", "skeleton_rig_legs", 
data_dir  = "./fast_cd_data/raw_data/" + name + "/" 
texturePath = './fast_cd_data/colormaps/RdBu_11.png' #_black.png' 
results_dir = "../results/skinning_mode_anim/" + name + "/"
bones = [0]
period = 48
amplitude = 1


mesh_file = data_dir + "/" + name + ".msh"
[V, F, T] = fcd.readMSH(mesh_file)
F = igl.boundary_facets(T)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))

mode_type = "skinning"
num_modes = 10
#ALSO  DO a Null Rig by default
rig_dir = results_dir +  "/null/"
try: 
    os.makedirs(rig_dir) 
except OSError as error: 
    print(error)
J = sp.sparse.csc_matrix((V.shape[0]*3, 0));
[B, Ws, L] = fcd.get_modes(V, T, np.array([[]]), J, mode_type, num_modes)
np.save(rig_dir + "V.npy", V)
np.save(rig_dir + "F.npy", F)
np.save(rig_dir + "Ws.npy", Ws)
np.save(rig_dir + "B.npy", B)

for rig_name in rig_names:
  rig_dir = data_dir +  "/rigs/" + rig_name + "/"
  rig_path = rig_dir + rig_name + ".json"
  f = open(rig_path)
  j = json.load(f)
  W = np.array(j["W"])
  Vs = np.array(j["V"])
  bl = np.array(j["lengths"])/so

  # P0 = np.array(j["p0"], order="F").reshape((W.shape[1]*4, 3))

  A = np.array(j["p0"]).transpose([0, 2, 1])
  P0 = A.reshape((W.shape[1]*4, 3))
  [W, P0] = fcd.fit_rig_to_mesh(W, P0, Vs, V, T);
  p0 = P0.flatten(order="F")

  rig_thickness = 0.5
  [rigV, rigF, rigC] = fcd.get_skeleton_mesh(rig_thickness, p0, bl )


  J = fcd.lbs_jacobian(V, W);
  [B, Ws, L] = fcd.get_modes(V, T, W, J, mode_type, num_modes)

  try: 
    os.makedirs(results_dir + "/" + rig_name + "/") 
  except OSError as error: 
    print(error)
  np.save(results_dir + "/" + rig_name + "/" + "V.npy", V)
  np.save(results_dir + "/" + rig_name + "/" + "F.npy", F)
  np.save(results_dir + "/" + rig_name + "/" + "Ws.npy", Ws)
  np.save(results_dir + "/" + rig_name + "/" + "B.npy", B)
  num_b = B.shape[1]/12
  assert(num_b >= np.max(np.array(bones)) and " Weight function doesn't have enough bones!" )

  for i in bones:
    #each bone i will have 12 columns in B
    bone_ind_in_B = np.array([4*i, 4*i + 1, 4*i + 2, 4*i + 3,  
                            4*i + num_b*4, 4*i + 1 + num_b*4, 4*i + 2+ num_b*4, 4*i + 3+ num_b*4,
                              4*i + num_b*8, 4*i + 1 + num_b*8, 4*i + 2+ num_b*8, 4*i + 3+ num_b*8] ).astype(np.int32);
    for ai in bone_ind_in_B:
      b = B[:, ai]
      for step in range(0, period):
        rad = step/period * np.pi * 2
        scale = amplitude * np.sin(rad  )
        D = scale * b.reshape((int(b.shape[0]/3), 3), order="F")
        U = V + D

