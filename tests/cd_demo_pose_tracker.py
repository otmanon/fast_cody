import cv2
import mediapipe as mp
import numpy as np
import scipy as sp
import os
import igl
import fast_cd_pyb as fcd
import pose_landmarks_to_positions 
import json

#mediapipe confiig for pose sceleton
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose 
headI = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rightArmI = [12, 14, 16, 18, 20, 22]
leftArmI = [11, 13, 15, 17, 19, 21]
bodI = [11, 12, 24, 23]
leftLegI = [24, 26, 28, 30, 32]
rightLegI = [23, 25, 27, 29, 31]

################## FAST CD CONFIG ###############

name = "charizard"
mesh_file = "./data/" + name + ".msh"
cache_dir = "./cache/" + name + "/"
meta_file = cache_dir + "/meta.json"
[V, F, T] = fcd.readMSH(mesh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
F = igl.boundary_facets(T);


############# FAST CD RIG #####################
rightArmInd = 2  # indeces in weight matrix corresponding to right Arm
leftArmInd = 3
rig_path = "./data/charizard_skeleton_rig_mp.json"
f = open(rig_path)
j = json.load(f)
W = np.array(j["W"])
Vs = np.array(j["V"])
P0 = np.array(j["p0"], order="F").reshape((6*4, 3))
[W, P0] = fcd.fit_rig_to_mesh(W, P0, Vs, V, T);
#w = W[:, rightArmInd];
#W = W[:, [rightArmInd, leftArmInd]]
J = fcd.lbs_jacobian(V, W);
##### FAST CD SIMULATION PARAMETERS
write_cache = True
read_cache = True
num_modes = 40
num_clusters = 200
num_clustering_features = 10
mu = 100
h = 1e-2
lam = 0
mode_type = 'skinning'
do_inertia = True

cache_params = {"num_modes" : num_modes, 
"num_clusters" : num_clusters, 
"mu" : mu, "h" : h, 
"lambda" : lam, 
"mode_type" : mode_type, 
"do_inertia" : do_inertia}
### Check cache for stored data
solver_params = fcd.cd_arap_local_global_solver_params(True, 10, 1e-3)
sim = fcd.fast_cd_arap_sim();
well_read = False
if read_cache and os.path.isfile(meta_file) :
    f = open(meta_file)
    data = json.load(f)
    if data == cache_params:
        sim  = fcd.fast_cd_arap_sim(cache_dir, solver_params)
        B2 = sim.params.B.copy();
        labels2  =  sim.params.labels.copy()
        sim.params = fcd.fast_cd_sim_params(V, T, B2, labels2, J, mu, lam, h, do_inertia) 
        well_read = True
if not well_read:
   # W2 = W[:, [leftArmInd, rightArmInd]];
    #J2 = fcd.lbs_jacobian(V, W2);
    [B, Ws, L] = fcd.get_modes(V, T, W, J, mode_type, num_modes)
    [labels, C] = fcd.compute_clusters(T, B, L, num_clusters, num_clustering_features)
    sim_params = fcd.fast_cd_sim_params(V, T, B, labels, J, mu, lam, h, do_inertia) 
    sim = fcd.fast_cd_arap_sim(sim_params, solver_params)
    if (write_cache):
        sim.save(cache_dir);
        with open(meta_file, 'w') as outfile:
                json.dump(cache_params, outfile, indent=2)


# set sim state 
z0 = np.zeros((num_modes*12, 1))
P = np.zeros((6, 4, 3), order="F");
P[:, :3, :3] = np.identity(3);
p = P.reshape(72, 1)
st = fcd.cd_sim_state(z0, z0, p, p)

# momentum leaking matrix
M = fcd.massmatrix(V, T)
D = fcd.momentum_leaking_matrix(V, T)
md = np.tile((M@ D).diagonal(), (3))
MD = sp.sparse.diags(md)
BMDJ = sim.params.B.T @  MD @ J


#init 3D libigl viewer
viewer = fcd.fast_cd_viewer()
viewer.set_mesh(V, F, 0)
viewer.invert_normals(True, 0)
color = np.array([144, 210, 236])/255.0
viewer.set_color(color, 0)
viewer.set_camera_zoom(2);
viewer.set_camera_eye( [2, 1, 5 ])

cap = cv2.VideoCapture(0)


callibrated = False
Dref = None
def callback():
  global J, callibrated, Dref, sim, st
    # For webcam input:
  with mp_pose.Pose(
  min_detection_confidence=0.5,
  min_tracking_confidence=0.5, model_complexity=0, smooth_landmarks = True) as pose:
    if cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        return

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
      image.flags.writeable = False
      image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      results = pose.process(image)
      if (results.pose_landmarks):

        D, vis = pose_landmarks_to_positions.pose_landmarks_to_numpy(results.pose_landmarks)
        D[:, 2] = 0
        if (np.all(vis[rightArmI] > 0.8)  and np.all(vis[leftArmI] > 0.8) and not (callibrated)):
          Dref =  D
          callibrated = True
        if (callibrated):
          #fit right arm rotation
          DrightArm =  D[rightArmI]
          DrefrightArm =  Dref[rightArmI];
          K = DrightArm - DrightArm.mean(axis=0);
          KX = DrefrightArm - DrefrightArm.mean(axis=0);
          [Rr, S] = igl.polar_dec(K.T @ KX);
          
          #fit left arm rotation
          DleftArm =  D[leftArmI]
          DrefleftArm =  Dref[leftArmI];
          K = DleftArm - DleftArm.mean(axis=0);
          KX = DrefleftArm - DrefleftArm.mean(axis=0);
          [Rl, S] = igl.polar_dec(K.T @ KX);

          #flatten this appropriately and fill with rotation
          P = np.zeros((6, 4, 3), order="F");
          P[:, :3, :3] = np.identity(3); #initialize all rig matrices to identity
          P[rightArmInd, :3, :3] = Rr.transpose()
          P[leftArmInd, :3, :3] = Rl.transpose()
          Ps = P.reshape((24, 3))
          p = Ps.reshape((72, 1), order="F")


          #get momentum leaking force/ boundary conditions
          f_ext = sim.params.invh2 * BMDJ @ (2.0 * st.p_curr.reshape((72, 1)) - st.p_prev.reshape((72, 1)) - p)
          bc = np.array([[]], dtype=np.float64).T
          #initial guess...  not sure if i shoulddelete the need of step to give as input z
          z = st.z_curr   
          z = sim.step(z, p,  st, f_ext, bc).reshape((12*num_modes, 1), order="F")
          st.update(z, p);

          U = np.reshape(J@p+ sim.params.B@z, (int(J.shape[0]/3), 3), order="F")
          viewer.set_vertices(U,  0 )
        # image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # mp_drawing.draw_landmarks(
        #     image,
        #     results.pose_landmarks,
        #     mp_pose.POSE_CONNECTIONS,
        #     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow('MediaPipe Pose', cv2.flip(image, 1))


viewer.set_pre_draw_callback(callback)
viewer.launch()

cap.release()