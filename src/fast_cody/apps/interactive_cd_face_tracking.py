import numpy as np
import os
from os.path import basename, splitext

import fast_cd_pyb as fcd
import fast_cody as fc


def interactive_cd_face_tracking(msh_file=None, V=None, T=None, Wp=None, bI=0, Ws=None, l=None, mu=1e4, rho=1e3,
                                 num_modes=16, num_clusters=100,
                                 constraint_enforcement="optimal",
                                 cache_dir=None, read_cache=False,
                                 texture_png=None, texture_obj=None,
                                 draw_landmarks=False):
    """
    Runs a standard interactive fast CD simulation, where the user can manipulate a single affine
    handle with a mediapipe face tracker in real time.


    Parameters
    ----------
    msh_file : str
        path to Tet mesh .msh file (usually generated by TetWild).
    V : float numpy array
        n x 3 vertex positions. If None, expects msh_file to be provided.
    T : int numpy array
        t x 4 tet indices. If None, expects msh_file to be provided.
    Wp : float numpy array
        n x m primary rig skinning weights used for control (default single handle full of ones)
    bI : int
        index in primary rig that we are controlling with our face
    Ws : float numpy array
        n x m skinning weights used for simulation. If None, recomputed on the fly
    l : int numpy array
        T x 1  per-tet cluster indices. if None, recomputed on the fly.
    num_modes : int
        if Ws is None, number of skinning modes to compute
    num_clusters : int
        if l is None, number of skinning clusters to compute.
    constraint_enforcement : str
        {"project", "optimal"}.
        if "optimal", performs the full constrained GEVP described in the paper.
    cache_dir : str
        directory where  cache is stored. if None, then
    read_cache : bool
        whether to read skinning modes from cache or not
    texture_obj : str
        directory pointing towards a .obj file of the surface mesh
        containing the UV map required for texture mapping.
        if None and if texture_png is None, then no texturing is applied. (default=None)
    texture_png : str
        directory pointing towards a .png file of the surface texture.
        if None and if texture_obj is None, then no texturing is applied. (default=None)
    draw_landmarks : bool
        whether to draw the face landmarks on the screen (default=False). This slows down the app if true


    Examples
    --------
    ```
    >>> import fast_cody as fcd
    >>> fcd.apps.interactive_cd_face_tracking()
    ```
    """

    if msh_file is not None:
        [V, F, T] = fcd.readMSH(msh_file)
    elif msh_file is None and (V is None and T is None):
        msh_file = fc.get_data("./cd_fish.msh")
        [V, F, T] = fcd.readMSH(msh_file)
    else:
        assert(V is not None and T is not None and "Must provide either msh_file or V and T")
    if cache_dir is None:
        cache_dir = "./cache/"
    os.makedirs(cache_dir, exist_ok=True)

    if texture_png is None or texture_obj is None:
        if msh_file ==  fc.get_data("./cd_fish.msh"):
            texture_png = fc.get_data("./cd_fish_tex.png")
            texture_obj = fc.get_data("./cd_fish_tex.obj")

    [V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))  # center to unit height and about origin

    if Wp is None:
        #assume affine handle
        Wp = np.ones((V.shape[0], 1))
    J = fc.lbs_jacobian(V, Wp)

    if Ws is None or l is None:
        C = fc.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fc.lbs_weight_space_constraint(V, C)
        [B, l, Ws] = fc.skinning_subspace(V, T, num_modes, num_clusters, C=C2, read_cache=read_cache,
                                          cache_dir=cache_dir, constraint_enforcement=constraint_enforcement);
    else:
        assert (Ws is not None and l is not None and "Secondary skinning weights and clusters need both be specified")
        num_modes = Ws.shape[1]
        num_clusters = l.max() + 1
    # cd_obj = fish_cd();
    sim = fc.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=1e-2, cache_dir="./results", read_cache=read_cache)


    # set sim initial state. z0 is full of 0, while p0 is the identity for all rig handles
    z0 = np.zeros((B.shape[1], 1))
    T0 = np.zeros((4, 3)).astype( dtype=np.float32)
    T0[0:3, 0:3] =  np.identity(3)
    T0 = np.tile(T0, (Wp.shape[1], 1))
    p0 = np.reshape( T0, (Wp.shape[1]*12, 1), order="F")


    st = fc.fast_cd_state(z0, p0)

    def user_callback():
        nonlocal T0

        [R, info] = face_captor.query_rotation()
        T0[4 * bI:4 * bI + 3, :] = R
        p = np.reshape(T0, ( J.shape[1], 1), order="F")
        z = sim.step(p,  st).reshape((B.shape[1], 1), order="F")
        st.update(z, p)
        viewer.update_subspace_coefficients(z, p)

        face_captor.imshow()


    viewer = fc.viewers.interactive_handle_subspace_viewer(V, T, Wp, Ws, user_callback,
                                                           texture_png=texture_png, texture_obj=texture_obj,
                                                           t0=to, s0=so, init_guizmo=False, max_fps=100)
    face_captor = fc.mediapipe_face_captor(draw_landmarks=draw_landmarks)

    viewer.launch()
    face_captor.release()
