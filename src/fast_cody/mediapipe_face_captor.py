import numpy as np
import mediapipe as mp
import cv2
import igl

from fast_cody import OneEuroFilter




#edge indices, libigl style
FE = np.array([[10, 338], [338, 297], [297, 332], [332, 284],
                                [284, 251], [251, 389], [389, 356], [356, 454],
                                [454, 323], [323, 361], [361, 288], [288, 397],
                                [397, 365], [365, 379], [379, 378], [378, 400],
                                [400, 377], [377, 152], [152, 148], [148, 176],
                                [176, 149], [149, 150], [150, 136], [136, 172],
                                [172, 58], [58, 132], [132, 93], [93, 234],
                                [234, 127], [127, 162], [162, 21], [21, 54],
                                [54, 103], [103, 67], [67, 109], [109, 10]])
def face_landmarks_to_positions(face_landmarks):
    num_V = mp.solutions.face_mesh.FACEMESH_NUM_LANDMARKS_WITH_IRISES
    V = np.zeros((num_V, 3))
    ind = 0
    for l in face_landmarks.landmark:
        V[ind, :] = np.array([l.x, l.y, l.z])
        ind += 1
    return V

class mediapipe_face_captor():
    """ Wrapper for mediapipe face capture.


    Example
    -------
    ```
    import fast_cody as fcd
    import time
    captor = fcd.mediapipe_face_captor()
    for i in range(1000):
        [R, info] = captor.query_rotation()
        captor.imshow()
        print(R)
    captor.release()
    ```
    """
    def __init__(self, R0=None, draw_landmarks=False):
        """
        Parameters
        ----------
        R0 : 3x3 float numpy array
            Initial rotation matrix used when face is not detected.
        draw_landmarks : bool
            Whether to draw landmarks on the image. Slows down the app if true
        """
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



    def query_rotation(s):
        """
        Returns
        -------
        R : 3x3 float numpy array
            Current rotation matrix.
        info : dict
            Dictionary containing 'image', 'multi_face_landmarks', and a 'calibrated' key.
        """
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
        """
        Displays the image on the screen. If self.draw_landmarks is True, also draws landmarks on the face.
        """
        if s.cap.isOpened():
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


            cv2.imshow('MediaPipe Face Detection', cv2.flip(s.image, 1))
            cv2.waitKey(1)
            # cv2.waitKey()
        # if (record):
        #     cv2.imwrite(results_dir + "/camera_stream/" + str(step).zfill(4) + ".png", image)
        #


    def release(self):
        """ Releases the camera.
        """
        self.cap.release()
