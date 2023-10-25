import numpy as np
import mediapipe as mp
import cv2
import igl

from fast_cd import OneEuroFilter, face_landmarks_to_positions


'''
This class wraps and organizes mediapipe to provide an interface that 
allows for easy access to facial quantities
'''
class mediapipe_face_captor():

    '''
    R0: initial rotation matrix (defualt identity(3))
    draw_landmarks: whether to draw landmarks on the image (default=False). Slows down the app if true
    '''

    def __init__(self, R0=None, draw_landmarks=False):
        if R0 is None:
            R0 = np.identity(3)
        self.R0 = R0
        self.draw_landmarks = draw_landmarks

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.cap = cv2.VideoCapture(0)
        self.calibrated = False  # used to pick rest positions used for rotaiton fitting
        self.PX = None  # initial landmark config filled up after first frame of detected face
        self.filter = OneEuroFilter(R0)

        self.face_control_power = 10

        # dynamic member variables whose quantities are changed throughout app
        self.image = None
        self.multi_face_landmarks = None


    '''
    Queries the current face capture rotation
    
    Returns:
        R: 3x3 current rotation matrix
        info - dict containing 'image', 'multi_face_landmarks', and a 'calibrated' key
    '''
    def query_rotation(s):
        R = s.R0
        with s.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

                if s.cap.isOpened():
                    success, image = s.cap.read()
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
                    s.image = image
                    s.multi_face_landmarks = results.multi_face_landmarks

                    if s.multi_face_landmarks:
                        P = face_landmarks_to_positions(s.multi_face_landmarks[0])
                        if not s.calibrated:
                            s.calibrated = True
                            s.PX = P - P.mean(axis=0)
                        K = P - P.mean(axis=0)
                        [R, S] = igl.polar_dec(K.T @ s.PX);
                        R = s.filter(R.T)

                info = {'calibrated': s.calibrated, 'landmarks':s.multi_face_landmarks, 'image':s.image}

                return R, info

    '''
    displays the image
    '''
    def imshow(s):
        if s.draw_landmarks:
            if s.multi_face_landmarks:
                for face_landmarks in s.multi_face_landmarks:
                    s.mp_drawing.draw_landmarks(
                        image=s.image,
                        landmark_list=face_landmarks,
                        connections=s.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=s.mp_drawing_styles
                        .get_default_face_mesh_tesselation_style())
                    s.mp_drawing.draw_landmarks(
                        image=s.image,
                        landmark_list=face_landmarks,
                        connections=s.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=s.mp_drawing_styles
                        .get_default_face_mesh_contours_style())
                    s.mp_drawing.draw_landmarks(
                        image=s.image,
                        landmark_list=face_landmarks,
                        connections=s.mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=s.mp_drawing_styles
                        .get_default_face_mesh_iris_connections_style())
    # Flip the image horizontally for a selfie-view display.

        cv2.imshow('MediaPipe Face Detection', cv2.flip(s.image, 1))
        # if (record):
        #     cv2.imwrite(results_dir + "/camera_stream/" + str(step).zfill(4) + ".png", image)
        #

    '''
    Called when closing the face captor
    '''
    def release(self):
        self.cap.release()