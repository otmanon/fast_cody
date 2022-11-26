import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np
import json
cwd = os.getcwd()


# User params
name = "charizard"
rig_names = ["null", "skeleton_rig_arms_legs"] #, "single_bone", "skeleton_rig_legs", 
data_dir  = "./data/" + name + "/" 
colormapPath = ['./data/colormaps/Paired_20.json', './data/colormaps/Paired_20.json']#_black.png' 
results_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/cluster_vis/" + name + "/"
cluster_count_list = [10, 20, 40, 80, 160]

imgRes_x = 800 
imgRes_y = 800 
numSamples = 10 
exposure = 1.5 

location = (0,0,0.67)
rotation = (90, 0,0) 
scale = (1,1,1)

c = 0
for rig_name in rig_names:
  print("###################### " + rig_name + "########################")
  rig_dir = results_dir + rig_name + "/"
  result_dir = results_dir + rig_name + "/clusters_png/";
  try: 
    os.mkdir(result_dir) 
  except OSError as error: 
    print(error)   

  V = np.load( rig_dir + "V.npy") # np.array([[1,1,1],[-1,1,-1],[-1,-1,1],[1,-1,-1]], dtype=np.float32) # vertex list
  print(V.shape)
  F=np.load(rig_dir + "F.npy")
  print(F.shape)

  ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
  camLocation = (1, -2, 0.75)
  lookAtLocation = (0, 0, 0.5)#(-0.5,-0.3,0.5)
  focalLength = 45 # (UI: click camera > Object Data > Focal Length)
  ### Set light
  lightAngle = (6, -30, -270) 
  strength = 2

  shadowSoftness = 0.3

  
  for i in range(len(cluster_count_list)):
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

    mesh = bt.readNumpyMesh(V,F,location,rotation,scale)
    ob = bpy.data.objects.get('numpy mesh object')

    #set lighting to smooth. 
    for poly in ob.data.polygons:
        poly.use_smooth = True

    outputPath = os.path.join(cwd, result_dir + '/cluster_' + str(i) + '.png') 
    
    clusters = np.load(rig_dir + "clusters_" + str(cluster_count_list[i]) + ".npy")
    face_scalars = clusters  # vertex color list
    
    #need to convert these to face colors... here's a ghetto way
    f = open(colormapPath[c])
    j = json.load(f)
    cmap = np.array(j['C']);
    print("Cmap rows :" , cmap.shape[0], "Cmap cols :" , cmap.shape[1])

    face_cind = (face_scalars / np.max(face_scalars) * (cmap.shape[0] -1)).astype( dtype=np.int32)
    face_colors = cmap[face_cind, :]  # face color list
    color_type = 'face'
    print("face_scalars", face_scalars.shape)
    print("face_colors", face_colors.shape)
    print("face_cind", face_cind.shape)
    print("#############################")
    mesh = bt.setMeshColors(mesh, face_colors, color_type)

    meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
    bt.setMat_VColor(mesh, meshVColor)

   # useless = (0,0,0,1)
    #meshColor = bt.colorObj(useless, 0.5,1, 1, 0.0, 0.0)
   # bt.setMat_texture(mesh, texturePath[c], meshColor)
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



Cdata = Ws[:, 0];
#Cdata = (Ws - Ws.mean());
maxim = np.abs(Cdata).max()
minim = np.abs(Cdata).max()

#Cdata = Cdata/(maxim);
Cdata = (Cdata) +  maxim;
Cdata = Cdata / (2*maxim);  
#min
print("min", Ws[:, 0].min())
#max
print("max", Ws[:, 0].max())


#min
print("min", Cdata.min())
#max
print("max", Cdata.max())