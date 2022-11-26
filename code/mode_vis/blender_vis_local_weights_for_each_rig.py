import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np

cwd = os.getcwd()


# User params
name = "charizard"
rig_names = ["skeleton_rig"] #, "single_bone", "skeleton_rig_legs", 
result_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/local_modes_visualization/charizard/"
texturePath = [ './data/colormaps/RdBu_11.png']#_black.png' 

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
  rig_dir = result_dir + rig_name + "/"
  try: 
    os.mkdir(result_dir) 
  except OSError as error: 
    print(error)   

  V = np.load( rig_dir + "V.npy") 
  print(V.shape)
  F=np.load(rig_dir + "F.npy")
  print(F.shape)
  Ws = np.load(rig_dir + "W.npy")
  print(Ws.shape)
  #max

  ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
  camLocation = (1, -2, 0.75)
  lookAtLocation = (0, 0, 0.5)#(-0.5,-0.3,0.5)
  focalLength = 45 # (UI: click camera > Object Data > Focal Length)
  ### Set light
  lightAngle = (6, -30, -270) 
  strength = 2

  shadowSoftness = 0.3
  for i in range(0, Ws.shape[1]):
    bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

    mesh = bt.readNumpyMesh(V,F,location,rotation,scale)
    ob = bpy.data.objects.get('numpy mesh object')

    #set lighting to smooth. 
    for poly in ob.data.polygons:
        poly.use_smooth = True

    outputPath = os.path.join(cwd, rig_dir + '/renders/secondary_weight_' + str(i) + '.png') 
    Cdata = Ws[:, i];

        #min
    print("min", Cdata.min())

    #max
    print("max", Cdata.max())
    #Cdata = (Ws - Ws.mean());
    maxim = np.abs(Cdata).max()

    #Cdata = Cdata/(maxim);
    Cdata = (Cdata) +  maxim;
    Cdata = Cdata / (2.0*maxim);  
    Cdata += 1e-3
    Cdata  = np.minimum(Cdata, 0.99)
    #min
    print("min", Cdata.min())

    #max
    print("max", Cdata.max())
    
   
    vertex_scalars = Cdata  # vertex color list
    color_type = 'vertex'
    color_map = 'default'
    print(vertex_scalars.shape)
    mesh = bt.vertexScalarToUV_unnormalized(mesh, vertex_scalars)

   # mesh = bt.setMeshScalars(mesh, vertex_scalars, color_map, color_type)
    
    useless = (0,0,0,1)
    meshColor = bt.colorObj(useless, 0.5,1, 1, 0.0, 0.0)
    bt.setMat_texture(mesh, texturePath[c], meshColor)
   # bt.setMat_VColor(mesh, meshColor)
 
    cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

    ## set light
    lightAngle = (6, -30, -270) 
    strength = 2
    shadowSoftness = 0.3
    sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

    # normalize cdata between [-1 1]
    


    ## set ambient light
    bt.setLight_ambient(color=(0.5,0.5,0.5,1)) 

    ## set gray shadow to completely white with a threshold 
    bt.shadowThreshold(alphaThreshold = 0.02, interpolationMode = 'CARDINAL')

    ## save blender file so that you can adjust parameters in the UI
    #bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')
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