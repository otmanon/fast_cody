import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np
from os import listdir

cwd = os.getcwd()


# User params

anim_dir = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/constrained_vs_unconstrained/elephant_high_res/rig/rig_mesh_recording/"
render_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/constrained_vs_unconstrained/elephant_high_res/rig/render_rig/"

anim_dir = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/constrained_vs_unconstrained/elephant_high_res/rig/mesh_recording/"
render_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/constrained_vs_unconstrained/elephant_high_res/rig/"

# anim_dir = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/constrained_vs_unconstrained/elephant_high_res/constrained_modes_unconstrained_sim/mesh_recording/"
# render_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/constrained_vs_unconstrained/elephant_high_res/constrained/render/"

# anim_dir = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/constrained_vs_unconstrained/elephant_high_res/unconstrained_modes_constrained_sim/elephant_high_res/mesh_recording/"
# render_dir = "C:/Users/otmanbench/Dropbox/fast-cd-results/experiments/constrained_vs_unconstrained/elephant_high_res/unconstrained/render/"
try: 
  os.makedirs(render_dir)
except OSError as error: 
  print(error)   

#make derek blue color tuple

# color = bt.derekBlue 
color = (0.75, 0.75, 0.75, 1.0)
# color = (153.0/255, 203.0/255, 67.0/255, 1) #eris lime green
# color =  (250/255., 114/255., 104/255., 1.0)
frame_ranges = [379, 380, 381, 382, 383, 384, 385, 386]
frame_ranges = [386]
imgRes_x = 800 
imgRes_y = 800 
numSamples = 100
exposure = 5

location = (0,0,0)
rotation = (90, 0,90) 
scale = (1,1,1)

c = 0
for i in frame_ranges:
  ### Set light
  lightAngle = (6, -30, -270) 
  strength = 2

  shadowSoftness = 0.3
  
  bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)

  mesh = bt.readOBJ(anim_dir + "/" + str(i).zfill(4) + ".obj", location, rotation, scale);
  bpy.ops.object.shade_smooth() 

 
  meshColor = bt.colorObj(color, 0.5, 1.0, 1.0, 0.0, 2.0)
  AOStrength = 0.0
  bt.setMat_balloon(mesh, meshColor, AOStrength)

  ## set invisible plane (shadow catcher)
  bt.invisibleGround(location=(0, 0, -0.55), shadowBrightness=1)

  ## set camera (recommend to change mesh instead of camera, unless you want to adjust the Elevation)
  camLocation = (2, 0, 0.1)
  lookAtLocation = (0,0,0.1)
  focalLength = 45 # (UI: click camera > Object Data > Focal Length)
  cam = bt.setCamera(camLocation, lookAtLocation, focalLength)

  ## set light
  lightAngle = (-70, 90, -60) 
  strength =1.0
  shadowSoftness = 1.0
  sun = bt.setLight_sun(lightAngle, strength, shadowSoftness)

  # lightAngle = (-70, 90, -150) 
  # strength = 0.0
  # shadowSoftness = 1.0
  # sun2 = bt.setLight_sun(lightAngle, strength, shadowSoftness)

  # lightAngle = (6, -30, 30) 
  # strength = 0.0
  # shadowSoftness = 0.3
  # sun4 = bt.setLight_sun(lightAngle, strength, shadowSoftness)

  ## set ambient light
  bt.setLight_ambient(color=(0.1,0.1,0.1,1)) 

  ## set gray shadow to completely white with a threshold 
  bt.shadowThreshold(alphaThreshold = 0.05, interpolationMode = 'CARDINAL')

  
  outputPath = os.path.join(cwd, render_dir +  "/" + str(i).zfill(4) + '.png') 
  bt.renderImage(outputPath, cam)
  c+=1

