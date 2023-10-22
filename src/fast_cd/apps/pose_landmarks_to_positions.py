import numpy as np
import mediapipe as mp
#Converts from a mediapipe face_landmarks datastructure to a simple numpy array
def pose_landmarks_to_positions(pose_landmarks):
    num_V = 33 # hard coded for mediapipes pose solution
    V = np.zeros((num_V, 3))
    l = pose_landmarks.landmark
    for i in range(0, num_V):
        V[i, :] = np.array([l[i].x, l[i].y, l[i].z])
  
    return V


def pose_landmarks_to_numpy(pose_landmarks):
    num_V = 33 # hard coded for mediapipes pose solution
    V = np.zeros((num_V, 3))
    vis = np.zeros((num_V, 1))
    l = pose_landmarks.landmark
    for i in range(0, num_V):
        V[i, :] = np.array([l[i].x, l[i].y, l[i].z])
        vis[i] = l[i].visibility
    return V, vis

