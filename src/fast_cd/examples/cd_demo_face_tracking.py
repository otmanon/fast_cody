import cv2
import mediapipe as mp
import fast_cd_pyb as fcd
import face_landmarks_to_positions 
import mediapipe_face
import igl
import numpy as np
import scipy as sp
import os
import json

#mediapipe preliminaries
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(1)
calibrated = False # used to pick rest positions used for rotaiton fitting
PX = None # filled up after first frame of detected face
face_control_scale = 10;

write_cache = True
read_cache = True

name = "bulldog"
mesh_file = "./fast_cd_data/raw_data/" + name + "/" + name + ".msh"
result_dir = "../results/cd_demo_face_tracking/"
cache_dir = result_dir + "./cache/" + name + "/"
meta_file = cache_dir + "/meta.json"
[V, F, T] = fcd.readMSH(mesh_file)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0,  0.]]))
F = igl.boundary_facets(T);

W = np.ones((V.shape[0], 1))
J = fcd.lbs_jacobian(V, W)

##### SIMULATION PARAMETERS
read_cache = True
write_cache = True
precomp_cache_dir = "./cache/"
modes_cache_dir = "./cache/"
clusters_cache_dir = "./cache/"

num_modes = 20
mode_type = "skinning"
subspace_constraint_type= "cd_momentum_leak"
num_clusters = 100
num_clustering_features = 10
split_components = True
mu = 10
h = 1e-2
lam = 0
mode_type = "skinning"
do_inertia = True


cache_params = {"num_modes" : num_modes, 
"num_clusters" : num_clusters, 
"mu" : mu, "h" : h, 
"lambda" : lam, 
"mode_type" : mode_type, 
"do_inertia" : do_inertia}
### Check cache for stored data
solver_params = fcd.cd_arap_local_global_solver_params(True, 100, 1e-6)

sub = fcd.fast_cd_subspace(num_modes, subspace_constraint_type, mode_type,
                           num_clusters, num_clustering_features, split_components)
sub.init_with_cache(V, T, J, read_cache, write_cache, modes_cache_dir, clusters_cache_dir, True, True)
labels = np.copy(sub.labels)  #need to copy this, for some reason its not writeable otherwise
B = np.copy(sub.B)
sim_params = fcd.fast_cd_sim_params(V, T, B, labels, J, mu, lam, h, do_inertia, "none")
sim = fcd.fast_cd_arap_sim(cache_dir, sim_params, solver_params, read_cache, write_cache)

sp = sim.sp();
dp = sim.dp();
solver = sim.sol();
    # if (write_cache):
    #     # sim.save(cache_dir);
    #     with open(meta_file, 'w') as outfile:
    #             json.dump(cache_params, outfile, indent=2)


# set sim state 
z0 = np.zeros((num_modes*12, 1))
T0 = np.identity(4).astype( dtype=np.float32, order="F");
p0 = T0[0:3, :].reshape( (12, 1))
st = fcd.cd_sim_state(z0, z0, p0, p0)

# momentum leaking matrix



#init 3D libigl viewer
viewer = fcd.fast_cd_viewer()
viewer.set_mesh(V, F, 0)
viewer.invert_normals(True, 0)
color = np.array([144, 210, 236])/255.0
viewer.set_color(color, 0)
def user_callback():
    global T0, J, calibrated, PX
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
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
                results = face_mesh.process(image)

                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.multi_face_landmarks:
                    P = face_landmarks_to_positions.face_landmarks_to_positions(results.multi_face_landmarks[0])
                    if not calibrated:
                        calibrated = True
                        PX = P - P.mean(axis=0)
                    K = P - P.mean(axis=0)
                    [R, S] = igl.polar_dec(K.T @ PX);
                    t = (P.mean(axis=0) - 0.5)*face_control_scale
                    t[0] *= -1
                    t[1]*= -1
                    t[2] = 0
                    T0[0:3, 3] =0*t
                    T0[0:3, 0:3] = R
                    p = T0[0:3, :].reshape( (12, 1))
                    f_ext = 0.0*np.zeros(st.z_curr.shape)
                    bc = np.array([[]], dtype=np.float64).T
                    #initial guess... 
                    z = st.z_curr   
                    z = sim.step(z, p,  st, f_ext, bc).reshape((12*num_modes, 1), order="F")
                    print(solver.prev_solve_iters)
                    st.update(z, p);
                    U = np.reshape(J@p+ B@z, (int(J.shape[0]/3), 3), order="F")
                    viewer.set_vertices(U, 0)
                    viewer.compute_normals(0)
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
                

viewer.set_pre_draw_callback(user_callback);

viewer.launch()

cap.release()
