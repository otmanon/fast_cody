import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np

cwd = os.getcwd()

name = "bar"
## initialize blender
imgRes_x = 800 
imgRes_y = 800 
numSamples = 200
exposure = 1.5 
## read mesh from numpy arrayss
location = (0,0,0)
rotation = (90, 0,0) 
scale = (0.2, 0.2,0.2)


result_dir = "./results/fit_modes/bar/";
try: 
  os.mkdir(result_dir) 
except OSError as error: 
  print(error)   

V = np.load( result_dir + "V.npy") # np.array([[1,1,1],[-1,1,-1],[-1,-1,1],[1,-1,-1]], dtype=np.float32) # vertex list
print(V.shape)
F=np.load(result_dir + "F.npy")
print(F.shape)
P = np.load( result_dir + "P.npy") 
print(P.shape)
P_lbs = np.load( result_dir + "P_lbs.npy") 
print(P_lbs.shape)
P_disp = np.load( result_dir + "P_disp.npy") 
print(P_disp.shape)
for method in ["rest", "ref", "lbs", "disp"]:
  print ("############## " + method)
  outputPath = os.path.join(cwd, result_dir + "/" + method +  '.png') 

  bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
  X = V;
  RGBA = (0.5, 0.5, 0.5, 1)
  if (method == "ref"):
    X = P
    RGBA = (173.0/255, 221.0/255, 142.0/255, 1)
  if (method == "lbs"):
    X = P_lbs
    RGBA = (144.0/255, 210.0/255, 236.0/255, 1)

  elif (method == "disp"):
    X = P_disp
    RGBA = (250/255, 114.0/255, 104.0/255, 1)
 
  mesh = bt.readNumpyMesh(X,F,location,rotation,scale)
  ob = bpy.data.objects.get('numpy mesh object')

  #set lighting to smooth. 
  # for poly in ob.data.polygons:
  #    poly.use_smooth = True


  # mesh = bt.setMeshScalars(mesh, vertex_scalars, color_map, color_type)
  
 
  meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
  bt.setMat_plastic(mesh, meshColor)

  ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
  camLocation = (0, -2, 0)
  lookAtLocation = (0, 0, 0)#(-0.5,-0.3,0.5)
  focalLength = 45 # (UI: click camera > Object Data > Focal Length)
  cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

  ## set light
  lightAngle = (6, -30, -45) 
  strength = 1
  shadowSoftness = 1
  sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

  ## set ambient light
  bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

  ## set gray shadow to completely white with a threshold 
  bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

  ## save blender file so that you can adjust parameters in the UI
  #bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

  # save rendering
  bt.renderImage(outputPath, cam)



