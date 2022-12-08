import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np

cwd = os.getcwd()

name = "spoon"
frame = 0

result_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/mode_fitting/" + name + "/"
npy_dir = result_dir + "/npy/" + str(frame) + "/"
## initialize blender
imgRes_x = 800 
imgRes_y = 800 
numSamples = 200
exposure = 1.0
## read mesh from numpy arrayss
location = (1,0.3, 0.25)
rotation = (90, -45, -45)
scale = ( 0.25,  0.25, 0.25)

# for twisting bar
# rotation = (0, 0, 90)
# scale = ( 0.25,  0.25, 0.25)
#for giraffe
# rotation = (90, 0,0) 
# scale = (1, 1, 1)



render_dir = result_dir + "/" +str(frame) +"/"
try: 
  os.mkdir(result_dir) 
except OSError as error: 
  print(error)   

V = np.load( npy_dir + "V.npy") # np.array([[1,1,1],[-1,1,-1],[-1,-1,1],[1,-1,-1]], dtype=np.float32) # vertex list
print(V.shape)
F=np.load(npy_dir + "F.npy")
print(F.shape)
P = np.load( npy_dir + "P.npy") 
print(P.shape)
P_lbs = np.load( npy_dir + "P_lbs.npy") 
print(P_lbs.shape)
P_disp = np.load( npy_dir + "P_disp.npy") 
print(P_disp.shape)
P_disp_derivs = np.load( npy_dir + "P_disp_derivs.npy") 
print(P_disp_derivs.shape)
for method in ["rest", "ref", "lbs", "disp", "derivs"]:
  print ("############## " + method)
  outputPath = os.path.join(cwd, render_dir +"/" + method +  '.png') 

  bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
  X = V;
  RGBA = (0.5, 0.5, 0.5, 1)
  RGBA = (173.0/255, 221.0/255, 142.0/255, 1)
  
  if (method == "ref"):
    X = P
    RGBA = (173.0/255, 221.0/255, 142.0/255, 1)
  
  if (method == "lbs"):
    X = P_lbs
    RGBA = (144.0/255, 210.0/255, 236.0/255, 1)

  elif (method == "disp"):
    X = P_disp
    RGBA = (250/255, 114.0/255, 104.0/255, 1)

  elif (method == "derivs"):
    X = P_disp_derivs
    RGBA = (201./255, 148.0/255, 199.0/255, 1)

  mesh = bt.readNumpyMesh(X,F,location,rotation,scale)
  # bpy.context.view_layer.update()
  ob = bpy.data.objects.get('numpy mesh object')
  ## set shading (uncomment one of them)
  # bpy.ops.object.shade_smooth() 
  bevel_mod = ob.modifiers.new(name="MY-Bevel2", type='BEVEL')
  bevel_mod.width = 1.0
  # ## subdivision
  bt.subdivision(mesh, level = 2)
  # bpy.ops.object.shade_smooth()  
  # #set lighting to smooth. 
  for poly in ob.data.polygons:
     poly.use_smooth = True


  # mesh = bt.setMeshScalars(mesh, vertex_scalars, color_map, color_type)
  # bt.invisibleGround(location=(0, 0, X[:, 1].min()), shadowBrightness=0.0)

  # bevel_mod = ob.modifiers.new(name="MY-Bevel", type='BEVEL')
  # bevel_mod.width = 1.0

  meshColor = bt.colorObj(RGBA, 0.5, 1.3, 1.0, 0.4, 2.0)
  AOStrength = 100
  bt.setMat_balloon(mesh, meshColor, AOStrength)

  ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
  camLocation = (-2, 0, 0)
  lookAtLocation = (0, 0, 0)#(-0.5,-0.3,0.5)
  focalLength = 45 # (UI: click camera > Object Data > Focal Length)
  cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

  ## set light
  lightAngle = (6, -30, -90) 
  strength = 0.0
  shadowSoftness = 0.01
  sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
  
  lightAngle = (0, -45, 0) 
  strength = 1
  shadowSoftness = 0.3
  sun2 = bt.setLight_sun(lightAngle, strength, shadowSoftness)

  ## set ambient light
  # bt.setLight_ambient(color=(0.5,0.5,0.5,1)) 

  ## set gray shadow to completely white with a threshold 
  bt.shadowThreshold(alphaThreshold = 0.0, interpolationMode = 'CARDINAL')

  ## save blender file so that you can adjust parameters in the UI
  #bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
  # bt.invisibleGround(location=(0, 0, scale[2]*np.min(X[:, 1]) + location[2]), shadowBrightness=0.9)

  # save rendering
  bt.renderImage(outputPath, cam)



