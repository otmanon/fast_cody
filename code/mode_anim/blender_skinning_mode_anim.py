import sys
sys.path.append('C:\\Users\\otmanbench\\Desktop\\utils\\BlenderToolbox\\') # change this to your path to “path/to/BlenderToolbox/
import BlenderToolBox as bt
import os, bpy, bmesh
import numpy as np

cwd = os.getcwd()

# User params
name = "charizard"
rig_names = [ "skeleton_rig_arms_legs"] #"null", "single_bone", "skeleton_rig_legs", 
data_dir  = "./data/" + name + "/"
texturePath = ['./data/colormaps/RdBu_11.png' ] #_black.png' 
results_dir = "./results/mode_anim/"

bones = [1]

period = 48
amplitude = 0.1
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
  Ws = np.load(rig_dir + "Ws.npy")
  B = np.load(rig_dir + "B.npy")
  num_b = B.shape[1]/12
  assert(num_b >= np.max(np.array(bones)) and " Weight function doesn't have enough bones!" )

  for i in bones:
    #each bone i will have 12 columns in B
    bone_ind_in_B = np.array([4*i, 4*i + 1, 4*i + 2, 4*i + 3,  
                            4*i + num_b*4, 4*i + 1 + num_b*4, 4*i + 2+ num_b*4, 4*i + 3+ num_b*4,
                              4*i + num_b*8, 4*i + 1 + num_b*8, 4*i + 2+ num_b*8, 4*i + 3+ num_b*8] ).astype(np.int32);
    affine_mode = 0
    for ai in bone_ind_in_B:
      b = B[:, ai]
      for step in range(0, period):
        rad = step/period * np.pi 
        s = amplitude * np.sin(rad  )
        D = s * b.reshape((int(b.shape[0]/3), 3), order="F")
        X = V + D
        anim_dir = output_dir +  'mode_anim_bone_' + str(i) + "/" + 'mode_' + str(affine_mode) + "/"

        try: 
            os.mkdir(anim_dir) 
        except OSError as error: 
            print(error)   

        outputPath = os.path.join(cwd, anim_dir + str(step).zfill(4) + '.png') 
       
        bt.blenderInit(imgRes_x, imgRes_y, numSamples, exposure)
        mesh = bt.readNumpyMesh(X,F,location,rotation,scale)
        ob = bpy.data.objects.get('numpy mesh object')
        #set lighting to smooth. 
        for poly in ob.data.polygons:
            poly.use_smooth = True

        Cdata = Ws[:, i];
        maxim = np.abs(Cdata).max();
        Cdata = (Cdata) +  maxim;
        Cdata = Cdata / (2*maxim);  
        Cdata += 1e-3
        vertex_scalars = Cdata;
        color_type = 'vertex'
        color_map = 'default'
        mesh = bt.vertexScalarToUV_unnormalized(mesh, vertex_scalars)

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
      affine_mode += 1
  c += 1





