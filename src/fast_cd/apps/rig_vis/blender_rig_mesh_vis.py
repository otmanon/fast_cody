import sys, os
# sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
sys.path.append("/Users/zhangjiayi/Documents/utils/BlenderToolbox") # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import bpy, bmesh
import numpy as np
cwd = os.getcwd()

outputPath = os.path.join(cwd, './results/demo_vertexColors.png') # make it abs path for windows

# User params
name = "charizard"
rig_name = "skeleton_rig_arms_legs"
data_dir  = "../../data/" + name + "/" 
rig_dir = data_dir + rig_name + "/"

## initialize blender
imgRes_x = 480 
imgRes_y = 480 
numSamples = 100 
exposure = 1.5 
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

## read mesh from numpy array
location = (0,0,0.67)
rotation = (90,0,0) 
scale = (1,1,1)

## read mesh (choose either readPLY or readOBJ)
meshPath = rig_dir + "rig_mesh.obj"
mesh = bt.readMesh(meshPath, location, rotation, scale)

# set shading (uncomment one of them)
bpy.ops.object.shade_smooth() 

## subdivision
# bt.subdivision(mesh, level = 0)

# # set material 
meshVColor = bt.colorObj([], 0.5, 1.0, 1.0, 0.0, 0.0)
bt.setMat_VColor(mesh, meshVColor)

## set invisible plane (shadow catcher)
bt.invisibleGround(shadowBrightness=0.9)

## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
camLocation = (1, -2, 0.75)
lookAtLocation = (0, 0, 0.5)#(-0.5,-0.3,0.5)
focalLength = 45 # (UI: click camera > Object Data > Focal Length)
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

### Set light
lightAngle = (6, -30, -270) 
strength = 2

shadowSoftness = 0.3

sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

## set gray shadow to completely white with a threshold 
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

## save blender file so that you can adjust parameters in the UI
bpy.ops.wm.save_mainfile(filepath=os.getcwd() + '/test.blend')

# save rendering
bt.renderImage(outputPath, cam)
