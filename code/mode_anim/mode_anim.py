import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import os
import numpy as np
import polyscope as ps

cwd = os.getcwd()


# User params
name = "charizard"
rig_names = ["skeleton_rig_arms_legs"] #"null", "single_bone", "skeleton_rig_legs", 
data_dir  = "./data/" + name + "/"
texturePath = './data/colormaps/RdBu_10.png' #_black.png' 
results_dir = "./results/mode_anim/"


bones = [0, 1, 2]

period = 48
amplitude = 1


for rig_name in rig_names:
  print("###################### " + rig_name + "########################")
  rig_dir = data_dir + rig_name + "/"

  output_dir = results_dir + name + '/' + rig_name + '/'
  try: 
    os.mkdir(output_dir) 
  except OSError as error: 
    print(error)   

  V = np.load( rig_dir + "V.npy") # np.array([[1,1,1],[-1,1,-1],[-1,-1,1],[1,-1,-1]], dtype=np.float32) # vertex list
  print(V.shape)
  F=np.load(rig_dir + "F.npy")
  print(F.shape)
  Ws = np.load(rig_dir + "Ws.npy")
  B = np.load(rig_dir + "B.npy")
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

