import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np

cwd = os.getcwd()


# User params
name = "charizard_disp"
rig_names = ["null", "skeleton_rig_arms_legs"] #"null", "single_bone", "skeleton_rig_legs", 
data_dir  = "./data/" + name + "/"
texturePath = ['./data/colormaps/RdBu_11.png', './data/colormaps/PRGn_11.png' ] #_black.png' 
results_dir = "./results/mode_anim/"


num_modes = 12;

period = 48
amplitude = 0.01
imgRes_x = 800 
imgRes_y = 800 
numSamples = 5 
exposure = 1.5 

location = (0,0,0.67)
rotation = (90, 0,0) 
scale = (0.7, 0.7, 0.7)

camLocation = (1, -2, 0.75)
lookAtLocation = (0, 0, 0.5)#(-0.5,-0.3,0.5)
focalLength = 45 # (UI: click camera > Object Data > Focal Length)
### Set light
lightAngle = (6, -30, -270) 
strength = 2
shadowSoftness = 0.3

c = 0
for rig_name in rig_names:
  print("###################### " + rig_name + "########################")
  rig_dir = data_dir + rig_name + "/"

  output_dir = results_dir + name + '/' + rig_name + '/'
  
  V = np.load( rig_dir + "V.npy") # np.array([[1,1,1],[-1,1,-1],[-1,-1,1],[1,-1,-1]], dtype=np.float32) # vertex list
  print(V.shape)
  F=np.load(rig_dir + "F.npy")
  print(F.shape)
  B = np.load(rig_dir + "B.npy")
  print(B.shape)
  for i in range(num_modes):
    for step in range(0, period):
      rad = step/period * np.pi 
      s = amplitude * np.sin(rad  )
      b = B[:, i]
      D = s * b.reshape((int(b.shape[0]/3), 3), order="F")
      X = V + D
      anim_dir = output_dir +   'mode_' + str(i) + "/"

      try: 
          os.makedirs(anim_dir) 
      except OSError as error: 
          print(error)   

      outputPath = os.path.join(cwd, anim_dir + str(step).zfill(4) + '.png') 
      
      bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
      mesh = bt.readNumpyMesh(X,F,location,rotation,scale)
      ob = bpy.data.objects.get('numpy mesh object')
      #set lighting to smooth. 
      for poly in ob.data.polygons:
          poly.use_smooth = True

      print(b.shape)
      D = b.reshape((int(b.shape[0]/3), 3), order="F");
      print(D.shape)
      vertex_scalars = np.linalg.norm(D, axis = 1);
      print(vertex_scalars.shape)
      print(V.shape)
        # vertex color list
      color_type = 'vertex'
      color_map = 'default'
      mesh = bt.vertexScalarToUV(mesh, vertex_scalars)

      # mesh = bt.setMeshScalars(mesh, vertex_scalars, color_map, color_type)
      
      useless = (0,0,0,1)
      meshColor = bt.colorObj(useless, 0.5, 1, 1, 0, 0)
      bt.setMat_texture(mesh, texturePath[c], meshColor)
      # bt.setMat_VColor(mesh, meshColor)
  
      cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

      ## set light
      lightAngle = (6, -30, -270) 
      strength = 2
      shadowSoftness = 0.3
      sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

      ## set ambient light
      bt.setLight_ambient(color=(0.5,0.5,0.5,1)) 

      ## set gray shadow to completely white with a threshold 
      bt.shadowThreshold(alphaThreshold = 0.02, interpolationMode = 'CARDINAL')

      ## save blender file so that you can adjust parameters in the UI
      bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

      # save rendering
      bt.renderImage(outputPath, cam)
  c += 1





