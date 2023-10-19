import numpy as np
import mediapipe as mp
#Converts from a mediapipe face_landmarks datastructure to a simple numpy array
def face_landmarks_to_positions(face_landmarks):
    num_V = mp.solutions.face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES
    V = np.zeros((num_V, 3))
    ind = 0
    for l in face_landmarks.landmark:
        V[ind, :] = np.array([l.x, l.y, l.z])
        ind += 1
    return V