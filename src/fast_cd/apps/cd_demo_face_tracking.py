import cv2
import mediapipe as mp
import fast_cd_pyb as fcdp
import fast_cd as fcd
import igl
import numpy as np
import os
from .fish_cd import fish_cd
from fast_cd import OneEuroFilter, face_landmarks_to_positions


def interactive_cd_face_tracking():
    #mediapipe preliminaries
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    calibrated = False # used to pick rest positions used for rotaiton fitting
    PX = None # filled up after first frame of detected face
    face_control_scale = 10;

    max_steps = 200
    record_mesh = False

    ##### SIMULATION PARAMETERS


    # cd_obj = hippo_cd()
    # cd_obj = bulldog_cd()
    cd_obj = fish_cd();

    # set sim state
    z0 = np.zeros((cd_obj.B.shape[1], 1))
    T0 = np.zeros((4, 3)).astype( dtype=np.float32)
    T0[0:3, 0:3] =  np.identity(3)

    T0 = np.tile(T0, (cd_obj.Wp.shape[1], 1))
    p0 = np.reshape( T0, (cd_obj.J.shape[1], 1), order="F")
    st = fcd.fast_cd_state(z0, p0)

    #init 3D libigl viewer
    viewer = fcdp.fast_cd_viewer_custom_shader(fcd.get_shader("vertex_shader_16.glsl"),
                                              fcd.get_shader("fragment_shader.glsl"), 16, 16)

    [Vf, TC, N, Ff, FTC, FN] = fcdp.readOBJ_tex(cd_obj.texture_obj);
    Vf = Vf * cd_obj.so - cd_obj.to
    Pr = fcdp.prolongation(Vf, cd_obj.V, cd_obj.T)

    viewer.set_mesh(Vf, Ff, 0)
    viewer.set_texture(cd_obj.texture_png, TC, FTC, 0)
    viewer.set_show_lines(False, 0)
    W_tex = Pr @ cd_obj.Wp;
    Ws_tex = Pr @ cd_obj.Ws;
    viewer.set_weights(W_tex, Ws_tex, 0)

    Zrec = np.zeros((z0.shape[0], 0))
    Prec = np.zeros((p0.shape[0], 0))

    filter = OneEuroFilter(p0)
    def write_mesh_recordings(Z, P):
       u = (cd_obj.B @ Z + cd_obj.J @ P)
       for i in range(Z.shape[1]):
             U =  Pr @np.reshape(u[:, i], (u[:, i].shape[0]//3, 3), order="F")
             os.makedirs(cd_obj.result_dir + "/mesh_recordings/", exist_ok=True)
             fcdp.writeOBJ_tex(cd_obj.result_dir + "/mesh_recordings/" + str(i) + ".obj", U, Ff, N, FN, TC, FTC)
       return

    os.makedirs(cd_obj.result_dir +  "/camera_stream/", exist_ok=True)
    step = 0

    def user_callback():
        nonlocal T0, calibrated, PX, Zrec, Prec, step
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
                        P = face_landmarks_to_positions(results.multi_face_landmarks[0])
                        if not calibrated:
                            calibrated = True
                            PX = P - P.mean(axis=0)
                        K = P - P.mean(axis=0)
                        [R, S] = igl.polar_dec(K.T @ PX);
                        t = (P.mean(axis=0) - 0.5)*face_control_scale

                        T0[4*cd_obj.bI:4*cd_obj.bI + 3, :] = R.T
                        # T0 = np.tile(T0, (W.shape[1], 1))
                        p = np.reshape(T0, ( cd_obj.J.shape[1], 1), order="F")
                        p = filter(p)

                        #initial guess...
                        z = st.z_curr
                        z = cd_obj.sim.step(p,  st).reshape((cd_obj.B.shape[1], 1), order="F")
                        # print(solver.prev_solve_iters)
                        st.update(z, p);

                        viewer.set_bone_transforms(p, z, 0);
                        viewer.updateGL(0)


                        if (record_mesh):
                            Zrec = np.concatenate((Zrec, z), axis=1)
                            Prec = np.concatenate((Prec, p), axis=1)
                            if (step == max_steps):
                                write_mesh_recordings(Zrec, Prec)

                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())
                            mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                            mp_drawing.draw_landmarks(
                                image=image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_IRISES,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_iris_connections_style())
                    # Flip the image horizontally for a selfie-view display.
                    step += 1
                    cv2.imshow('MediaPipe Face Detection', cv2.flip(image, 1))
                    if (record_mesh):
                        cv2.imwrite(cd_obj.result_dir + "/camera_stream/" + str(step).zfill(4) + ".png", image)
        viewer.updateGL(0)

    viewer.set_pre_draw_callback(user_callback);

    viewer.launch(90, True)

    print("num verts: ", cd_obj.V.shape[0])
    print("num tets: ", cd_obj.T.shape[0])
    cap.release()
