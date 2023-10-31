import igl
import numpy as np
import os
from os.path import basename, splitext

import fast_cd_pyb as fcdp
import fast_cody as fcd

'''
Runs a standard interactive fast CD simulation, where the user can play back rig animations
and observe secondary effects in real-time


Optional:
    msh_file: path to Tet mesh .msh file (usually generated by TetWild) default=None, which calls the cd_fish.msh included with the package
    rig_file: path to rig.json file (usually generated from Blender/Mixamo + riggrats), default=None, which assumes single affine handle
    anim_file: path to anim.json file (usually generated from Blender/Mixamo + riggrats), default=None, which assumes no motion for 1000 timesteps
    Wp - n x m primary rig skinning weights used for control (default=None is single handle full of ones). Ignored if rig.json is specified. Must have same
                number of handles as specified in anim_file.
    Ws - n x m skinning weights used for simulation. (default=None) If None, recomputed on the fly.
    l - T x 1  per-tet cluster indices. (default=None) if None, recomputed on the fly. 
    num_modes - if Ws is None, number of skinning modes to compute. (default=16) Ignored if Ws and l are passed.
    num_clusters - if l is None, number of skinning clusters to compute. (default=100) Ignored if Ws and l are passed.
    constraint_enforcements - {"project", "optimal"}. 
                                if "optimal", performs the full constrained GEVP described in the paper. 
                                                This is very slow in python for now, and much faster in Matlab
                                if "project", does an unsconstrained GEVP and projects to the CD constraint set. 
                                default = "project"
    results_dir - directory where results are stored and where cache is stored. if None, then
                    results_dir = "./results/interactive_cd_affine_handle/<msh_file_stem>/
    read_cache - whether to read skinning modes from cache or not (default=False)
    texture_obj - directory pointing towards a .obj file of the surface mesh 
                    containing the UV map required for texture mapping.
                    if None and if texture_png is None, then no texturing is applied. (default=None)
    texture_png - directory pointing towards a .png file of the surface texture.
                  if None and if texture_obj is None, then no texturing is applied. (default=None)      
    '''
def interactive_cd_rig_anim(msh_file=None, rig_file=None, anim_file=None, Ws=None, l=None,
                                 mu=1e4, rho=1e3, num_modes=16, num_clusters=100,
                                 constraint_enforcement="optimal",
                                 results_dir=None, read_cache=False,
                                 texture_png=None, texture_obj=None):
    """
    Runs a standard interactive fast CD simulation, where the user can play back rig animations
    and observe secondary effects in real-time

    Parameters
    ----------
    msh_file : str
        path to Tet mesh .msh file (usually generated by TetWild) default=None, which calls the cd_fish.msh included with the package
    rig_file : str
        path to rig.json file (usually generated from Blender/Mixamo + riggrats), default=None, which assumes single affine handle
    anim_file : str
        path to anim.json file (usually generated from Blender/Mixamo + riggrats), default=None, which assumes no motion for 1000 timesteps
    Ws : float numpy array
        n x m skinning weights used for simulation.  If None, recomputed on the fly.
    l : float int array
        T x 1  per-tet cluster indices.  if None, recomputed on the fly.
    mu : float
        mesh lame parameter
    rho : float
        mesh density
    num_modes : int
        if Ws is None, number of skinning modes to compute.Ignored if Ws and l are passed.
    num_clusters : int
        if l is None, number of skinning clusters to compute. Ignored if Ws and l are passed.
    constraint_enforcement : str
        {"project", "optimal"}.
        if "optimal", performs the full constrained GEVP described in the paper.
    results_dir : str
        directory where results are stored and where cache is stored. if None, then
    read_cache : bool
        whether to read skinning modes from cache or not (default=False)
    texture_obj : str
        directory pointing towards a .obj file of the surface mesh
        containing the UV map required for texture mapping. If None and if texture_png is None, then no texturing is applied.
    texture_png : str
        directory pointing towards a .png file of the surface texture.
        if None and if texture_obj is None, then no texturing is applied.


    Examples
    --------
    >>> import fast_cody as fcd
    >>> fcd.apps.interactive_cd_rig_anim()
    """

    if msh_file is None:
        msh_file = fcd.get_data("./cd_fish.msh")
        if texture_png is None or texture_obj is None:
            texture_png = fcd.get_data("./cd_fish_tex.png")
            texture_obj = fcd.get_data("./cd_fish_tex.obj")
        if rig_file is None:
            rig_file = fcd.get_data("./cd_fish_rig.json")
        if anim_file is None:
            anim_file = fcd.get_data("./cd_fish_rig_anim__swim.json")

    assert (splitext(msh_file)[1] == '.msh' and "only supports .msh file format")
    name = basename(splitext(msh_file)[0])
    msh_file = msh_file
    if results_dir is None:
        results_dir = "./results/interactive_cd_rig_anim/" + name + "/"
    cache_dir = results_dir + "/cache/"
    os.makedirs(cache_dir, exist_ok=True)
    [V, F, T] = fcdp.readMSH(msh_file)

    if rig_file is not None:
        [Vpsurf, Fpsurf, Wpsurface, P0, lengths, pI] = fcd.read_rig_from_json(rig_file)
        aI = np.arange(V.shape[0])
        [D2, bI, CP] = igl.point_mesh_squared_distance(Vpsurf, V, aI)
        Wp = fcd.diffuse_weights(V, T, Wpsurface, bI, dt=10000)
        # fcd.WeightsViewer(V, T, Wp, period=0.01)
    else:
        Wp = np.ones((V.shape[0], 1))
    [V, so, to] = fcdp.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))  # center to unit height and about origin
    # so = 1
    # to = 0
    P0= P0 * so
    P0[:, :, 3] = P0[:, :, 3] - to
    if Wp is None and rig_file is None:
        #assume affine handle
        Wp = np.ones((V.shape[0], 1))


    J = fcd.lbs_jacobian(V, Wp)

    if anim_file is None:
        print("Haven't handled anim_file is None yet")
    else:
        P = fcd.read_rig_anim_from_json(anim_file)

    P = P * so
    P[:, :, :, 3] = P[:, :, :, 3] - to
    Prel = fcd.world2rel(P, P0)

    d = V.shape[1]
    k = Wp.shape[1]
    frames = P.shape[0]
    Prel = np.transpose(Prel, [3, 1, 2, 0])
    Prel = Prel.reshape(((d + 1) * d * k, frames), order='F')


    if Ws is None or l is None:
        C = fcd.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fcd.lbs_weight_space_constraint(V, C)
        [B, l, Ws] = fcd.skinning_subspace(V, T, num_modes, num_clusters, C=C2, read_cache=read_cache,
                                          cache_dir=cache_dir, constraint_enforcement=constraint_enforcement);
    else:
        assert (Ws is not None and l is not None and "Secondary skinning weights and clusters need both be specified")
        num_modes = Ws.shape[1]
        num_clusters = l.max() + 1
    # cd_obj = fish_cd();
    sim = fcd.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=1e-2, cache_dir="./results", read_cache=read_cache)

    # set  sim initial state. z0 is full of 0, while p0 is the identity for all rig handles
    z0 = np.zeros((B.shape[1], 1))
    p0 = Prel[:, [0]]
    st = fcd.fast_cd_state(z0, p0)
    step = 0
    def user_callback():
        nonlocal step, st

        if step % frames == 0:
            st = fcd.fast_cd_state(z0, p0)

        p = Prel[:, step % frames]
        z = sim.step(p,  st).reshape((B.shape[1], 1), order="F")
        st.update(z, p)
        viewer.update_subspace_coefficients(z, p)

        step += 1

    viewer = fcd.viewers.interactive_handle_subspace_viewer(V, T, Wp, Ws, user_callback,
                                                           texture_png=texture_png, texture_obj=texture_obj,
                                                           t0=to, s0=so, init_guizmo=False, max_fps=100)
    viewer.launch()
