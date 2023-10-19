import numpy as np
import igl
import fast_cd_pyb as fcd
import json
import polyscope as ps
name = "charizard"
rig_name = "single_bone"
rig_path = "../fast_cd_cpp/data/raw_data/"+ name +"/rigs/" + name +"/" + name +".json"
mesh_file = "../fast_cd_cpp/data/raw_data/"+ name +"/" + name + ".msh"
data_path = "./data/" + name +"/" 
############
############# FAST CD RIG #####################
rightArmInd = 2  # indeces in weight matrix corresponding to right Arm
leftArmInd = 3
f = open(rig_path)
j = json.load(f)
W = np.array(j["W"])
V = np.array(j["V"])
F = np.array(j["F"])

# need to transpose the last two indices
A = np.array(j["p0"]).transpose([0, 2, 1])
P0 = A.reshape((W.shape[1]*4, 3))
p0 = P0.flatten(order="F")
bl = np.ones(W.shape[1])

[rV, rF, rC] = fcd.get_skeleton_mesh(0.2, p0, bl )


ps.init()
ps.register_surface_mesh("mesh", V, F)
ps.register_surface_mesh("rig", rV, rF)

ps.show()
