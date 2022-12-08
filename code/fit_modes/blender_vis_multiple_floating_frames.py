import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np
import json

cwd = os.getcwd()

name = "bar"
frame = 8

result_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/mode_fitting/" + name + "/"
npy_dir = result_dir + "/npy/" + str(frame) + "/"
colormap_path = './data/colormaps/Paired_20.json'
## initialize blender
imgRes_x = 800 
imgRes_y = 800 
numSamples = 200
exposure = 3.0
## read mesh from numpy arrayss
location = (0,0, 0)
rotation = (90, 0, 90)
scale = ( 0.3,  0.3, 0.3)

# for twisting bar
# rotation = (0, 0, 90)
# scale = ( 0.25,  0.25, 0.25)
#for giraffe
# rotation = (90, 0,0) 
# scale = (1, 1, 1)



render_dir = result_dir + "/render/"
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
P_disp_multiple = np.load( npy_dir + "P_disp_1000_floating_frames.npy") 
print(P_disp_multiple.shape)
P_disp_frame = np.load( npy_dir + "P_disp_floating.npy") 
print(P_disp_frame.shape)
for method in [ "disp_multiple", "rest", "ref", "lbs", "disp_frame"]:
  print ("############## " + method)
  outputPath = os.path.join(cwd, result_dir + "/" + method +  '.png') 

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

  elif (method == "disp_frame"):
    X = P_disp_frame
    RGBA = (250/255, 114.0/255, 104.0/255, 1)

  elif (method == "disp_multiple"):
    X = P_disp_multiple
    RGBA = (250/255, 114.0/255, 104.0/255, 1)
 
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
  # bt.invisibleGround(location=(0, 0, X[:, 1].min()), shadowBrightness=0.9)

  if (method =="disp_multiple"):
    clusters = np.load(npy_dir + "face_clusters_1000.npy")
    face_scalars = clusters  # vertex color list
    
    #need to convert these to face colors... here's a ghetto way
    f = open(colormap_path, 'r')
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
  else:
    meshColor = bt.colorObj(RGBA, 0.5, 1.0, 1.0, 0.0, 2.0)
    AOStrength = 0
    bt.setMat_balloon(mesh, meshColor, AOStrength)


  ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
  camLocation = (-2, 0, 0)
  lookAtLocation = (0, 0, 0)#(-0.5,-0.3,0.5)
  focalLength = 45 # (UI: click camera > Object Data > Focal Length)
  cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

  # ## set light
  # lightAngle = (6, -30, -90) 
  # strength = 0.5
  # shadowSoftness = 0.01
  # sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
  
  lightAngle = (-10, -45, 0) 
  strength = 1
  shadowSoftness = 0.1
  sun2 = bt.setLight_sun(lightAngle, strength, shadowSoftness)

  ## set ambient light
  bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

  ## set gray shadow to completely white with a threshold 
  bt.shadowThreshold(alphaThreshold = 0.3, interpolationMode = 'CARDINAL')

  ## save blender file so that you can adjust parameters in the UI
  #bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
  bt.invisibleGround(location=(0, 0, scale[2]*np.min(X[:, 1]) + location[2]), shadowBrightness=0.9)

  # save rendering
  bt.renderImage(outputPath, cam)



