import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np
from os import listdir

cwd = os.getcwd()


# User params
surface_mesh_obj = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/constrained_vs_unconstrained/elephant_high_res/deformed_pose/mesh.obj"
rig_mesh_obj = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/constrained_vs_unconstrained/elephant_high_res/deformed_pose/rig_mesh.obj"
render_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/constrained_vs_unconstrained/elephant_high_res/deformed_pose/"

try: 
  os.makedirs(render_dir)
except OSError as error: 
  print(error)   

#make derek blue color tuple

mesh_color = (0.6, 0.6, 0.6,1) #eris grey
rig_color = (153.0/255, 203.0/255, 67.0/255, 1) #eris lime green

imgRes_x = 800 
imgRes_y = 800 
numSamples = 100
exposure = 5

location = (0,0,0)
rotation = (90, 0,-90) 
scale = (5, 5,5)
#camera params
camLocation = (-10, 0, 0.1)
lookAtLocation = (0,0,0.1)
focalLength = 45 # (UI: click camera > Object Data > Focal Length)

## set light
lightAngle = (-70, 90, -155) 
strength =1.0
shadowSoftness = 1.0


#render rest mesh first
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
mesh = bt.readOBJ(surface_mesh_obj, location, rotation, scale);
bpy.ops.object.shade_smooth() 
meshColor = bt.colorObj(mesh_color, 0.5, 1.0, 1.0, 0.0, 2.0)
AOStrength = 0.0
bt.setMat_balloon(mesh, meshColor, AOStrength)
## set ambient light
bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 
## set gray shadow to completely white with a threshold 
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')
outputPath = os.path.join(cwd, render_dir +  "/" + 'rest_mesh.png') 
bt.renderImage(outputPath, cam)


#render rig mesh second
bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
cam = bt.setCamera(camLocation, lookAtLocation, focalLength)
#sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
mesh = bt.readOBJ(rig_mesh_obj, location, rotation, scale);
bpy.ops.object.shade_smooth() 
meshColor = bt.colorObj(rig_color, 0.5, 1.0, 1.0, 0.0, 2.0)
AOStrength = 0.0
bt.setMat_balloon(mesh, meshColor, AOStrength)
## set ambient light
sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)
bt.setLight_ambient(color=(0.1, 0.1, 0.1,1)) 
## set gray shadow to completely white with a threshold 
bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')
outputPath = os.path.join(cwd, render_dir +  "/" + 'rest_rig.png') 
bt.renderImage(outputPath, cam)
